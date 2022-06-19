import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from nets.mobilenetv2 import InvertedResidual, mobilenet_v2
from nets.LRFE import LRFE
from nets.attention import ECA



class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma      = scale or None
        self.eps        = 1e-10
        self.weight     = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm    = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x       = torch.div(x,norm)
        out     = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=True, up_size=0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x


class LMSN300(nn.Module):
    def __init__(self, head, ft_module, pyramid_ext, num_classes, pretrained = False):
        super(LMSN300, self).__init__()
        self.num_classes    = num_classes

        self.ft_module = nn.ModuleList(ft_module)
        self.pyramid_ext = nn.ModuleList(pyramid_ext)
        self.fea_bn = nn.BatchNorm2d(256 * len(self.ft_module), affine=True)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.mobilenet = mobilenet_v2(pretrained).features
        self.L2Norm = L2Norm(32, 20)

        self.Norm1 = LRFE(32, 32, stride=1, scale=1.0)
        self.Norm2 = LRFE(96, 96, stride=1, scale=1.0)
        self.Norm3 = LRFE(320, 320, stride=1, scale=1.0)

        # ECA module 【6 feature layers：512 512 256 256 128 128 】
        self.ECA1 = ECA(512)
        self.ECA2 = ECA(512)
        self.ECA3 = ECA(256)
        self.ECA4 = ECA(256)
        self.ECA5 = ECA(128)
        self.ECA6 = ECA(128)

    def forward(self, x):
        #---------------------------#
        #   x 300,300,3
        #---------------------------#
        sources = list()
        transformed_features = list()
        loc     = list()
        conf    = list()

        # ---------------------------#
        #   mobilenet shape 38,38,32
        # ---------------------------#
        for k in range(7):
            x = self.mobilenet[k](x)

        s = self.Norm1(x)
        sources.append(s)

        # ---------------------------#
        #   mobilenet shape 19,19,96
        # ---------------------------#
        for k in range(7, 14):
            x = self.mobilenet[k](x)
        s = self.Norm2(x)
        sources.append(s)

        # ---------------------------#
        #   mobilenet shape 10,10,320
        # ---------------------------#
        for k in range(14, 18):
            x = self.mobilenet[k](x)
        s = self.Norm3(x)
        sources.append(s)

        assert len(self.ft_module) == len(sources)
        for k, v in enumerate(self.ft_module):
            transformed_features.append(v(sources[k]))
        # x = torch.cat(transformed_features, 1)
        concat_fea = torch.cat(transformed_features, 1)
        x = self.fea_bn(concat_fea)
        pyramid_fea = list()
        for k, v in enumerate(self.pyramid_ext):
            x = v(x)
            pyramid_fea.append(x)

        # pyramid_fea store the newly generated feature pyramid，
        # (38,38,512), (19,19,512),(10,10,256),(5,5,256),（3,3,128）， (1,1,128)
        # sources_final saves the final detection layers after adding the ECA module
        sources_final = list()

        sources_final.append(self.ECA1(pyramid_fea[0]))
        sources_final.append(self.ECA2(pyramid_fea[1]))
        sources_final.append(self.ECA3(pyramid_fea[2]))
        sources_final.append(self.ECA4(pyramid_fea[3]))
        sources_final.append(self.ECA5(pyramid_fea[4]))
        sources_final.append(self.ECA6(pyramid_fea[5]))

        #-------------------------------------------------------------#
        #   Add regression prediction and classification prediction for the 6 effective feature layers
        #-------------------------------------------------------------#
        for (x, l, c) in zip(sources_final, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #-------------------------------------------------------------#
        #   reshape
        #-------------------------------------------------------------#  
        loc     = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf    = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #-------------------------------------------------------------#
        #   loc will reshape to batch_size, num_anchors, 4
        #   conf will reshap to batch_size, num_anchors, self.num_classes
        #-------------------------------------------------------------#     
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )
        return output

def feature_transform_module():
    layers = []

    # conv6    38,38,32 -> 38,38,256
    layers += [BasicConv(32, 256, kernel_size=1, stride=1, padding=0)]
    # conv13  19,19,96 -> 38,38,256
    layers += [BasicConv(96, 256, kernel_size=1, stride=1, padding=0, up_size=38)]
    # conv17  10,10,320 -> 38,38,256
    layers += [BasicConv(320, 256, kernel_size=1, stride=1, padding=0, up_size=38)]
    return layers

def pyramid_feature_extractor():
    layers = [
              # 38,38,256*3 -> 38,38,512
              InvertedResidual(256 * 3, 512, stride=1, padding=1, expand_ratio=0.25),
              # 38,38,512 -> 19,19,512
              InvertedResidual(512, 512, stride=2, padding=1, expand_ratio=0.5),
              # 19,19,512 -> 10,10,256
              InvertedResidual(512, 256, stride=2, padding=1, expand_ratio=0.25),
              # 10,10,256 -> 5,5,256
              InvertedResidual(256, 256, stride=2, padding=1, expand_ratio=0.5),
              # 5,5,256 -> 3,3,128
              InvertedResidual(256, 128, stride=2, padding=1, expand_ratio=0.5),
              # 3,3,128 ->1,1,128
              InvertedResidual(128, 128, stride=2, padding=0, expand_ratio=0.5)]

    return layers

def get_LMSN(num_classes):
    mbox = [6, 6, 6, 6, 6, 6]
    fea_channels = [512, 512, 256, 256, 128, 128]

    loc_layers = []
    conf_layers = []

    for k, fea_channels in enumerate(fea_channels):
        loc_layers += [nn.Conv2d(fea_channels, mbox[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(fea_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]

    FSSD_MODEL = LMSN300((loc_layers, conf_layers), feature_transform_module(), pyramid_feature_extractor(), num_classes)

    return FSSD_MODEL