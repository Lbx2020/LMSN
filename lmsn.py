import colorsys
import os
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont

from nets.LMSN import get_LMSN
from utils.anchors import get_anchors
from utils.utils import cvtColor, get_classes, resize_image, preprocess_input
from utils.utils_bbox import BBoxUtility

warnings.filterwarnings("ignore")

class LMSN(object):
    _defaults = {
        "model_path"        : 'model_data/LMSN_weight_RSOD.pth',
        "classes_path"      : 'model_data/RSOD_classes.txt',

        "input_shape"       : [300, 300],

        "confidence"        : 0.5,

        "nms_iou"           : 0.45,

        # ----------------------------------------------------#
        #   pascal voc, 'anchors_size'      : [30, 60, 111, 162, 213, 264, 315],
        #   RSOD, 'anchors_size'      : [18, 36, 93, 150, 207, 264, 321],
        # ----------------------------------------------------#
        # 'anchors_size'      : [30, 60, 111, 162, 213, 264, 315],
        'anchors_size': [18, 36, 93, 150, 207, 264, 321],

        "letterbox_image"   : False,

        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors                        = torch.from_numpy(get_anchors(self.input_shape, self.anchors_size)).type(torch.FloatTensor)
        if self.cuda:
            self.anchors = self.anchors.cuda()
        self.num_classes                    = self.num_classes + 1
        

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility(self.num_classes)
        self.generate()

    #---------------------------------------------------#
    #   load model
    #---------------------------------------------------#
    def generate(self):
        self.net = get_LMSN(self.num_classes)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()


    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():

            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs     = self.net(images)

            results     = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)

            if len(results[0]) <= 0:
                return image

            top_label   = np.array(results[0][:, 4], dtype = 'int32')
            top_conf    = results[0][:, 5]
            top_boxes   = results[0][:, :4]

        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)
        

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):

        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():

            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs     = self.net(images)

            results     = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():

                outputs     = self.net(images)

                results     = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image, 
                                                        nms_iou = self.nms_iou, confidence = self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 

        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs     = self.net(images)

            results     = self.bbox_util.decode_box(outputs, self.anchors, image_shape, self.input_shape, self.letterbox_image, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)

            if len(results[0]) <= 0:
                return 

            top_label   = np.array(results[0][:, 4], dtype = 'int32')
            top_conf    = results[0][:, 5]
            top_boxes   = results[0][:, :4]
        
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 