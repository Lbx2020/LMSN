import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.LMSN import get_LMSN
from nets.LMSN_training import MultiboxLoss, weights_init
from utils.anchors import get_anchors
from utils.callbacks import LossHistory
from utils.dataloader import SSDDataset, ssd_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    Cuda = True

    classes_path    = 'model_data/RSOD_classes.txt'

    model_path      = 'model_data/mobilenetv2_ssd_weights.pth'

    input_shape     = [300, 300]

    pretrained      = False

    #----------------------------------------------------#
    #   pascal voc, anchors_size    = [30, 60, 111, 162, 213, 264, 315]
    #   RSOD, anchors_size = [18, 36, 93, 150, 207, 264, 321]
    #----------------------------------------------------#
    # anchors_size    = [30, 60, 111, 162, 213, 264, 315]
    anchors_size = [18, 36, 93, 150, 207, 264, 321]

    #----------------------------------------------------#
    #   The training is divided into two phases, the freezing phase and the unfreezing phase.
    #----------------------------------------------------#
    #----------------------------------------------------#
    #   Freeze phase training parameters
    #----------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 16
    Freeze_lr           = 5e-4
    #----------------------------------------------------#
    #   unfreezing phase training parameters
    #----------------------------------------------------#
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 8
    Unfreeze_lr         = 1e-4

    Freeze_Train        = True

    num_workers         = 4
    #----------------------------------------------------#
    #   get image path and label
    #----------------------------------------------------#
    train_annotation_path   = '2015_train.txt'
    val_annotation_path     = '2015_val.txt'

    #----------------------------------------------------#
    #   get classes and anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    num_classes += 1
    anchors = get_anchors(input_shape, anchors_size)

    model = get_LMSN(num_classes)

    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        a = {}
        for k, v in pretrained_dict.items():
            try:
                if np.shape(model_dict[k]) == np.shape(v):
                    a[k] = v
            except:
                pass
        model_dict.update(a)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    criterion       = MultiboxLoss(num_classes, neg_pos_ratio=3.0)
    loss_history    = LossHistory("logs/")

    #---------------------------#
    #   Read the txt corresponding to the dataset
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)


    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch

        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)

        train_dataset   = SSDDataset(train_lines, input_shape, anchors, batch_size, num_classes, train = True)
        val_dataset     = SSDDataset(val_lines, input_shape, anchors, batch_size, num_classes, train = False)

        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=ssd_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=ssd_dataset_collate)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")


        for param in model.mobilenet.parameters():
            param.requires_grad = False

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, criterion, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()

    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch

        optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.94)

        train_dataset   = SSDDataset(train_lines, input_shape, anchors, batch_size, num_classes, train = True)
        val_dataset     = SSDDataset(val_lines, input_shape, anchors, batch_size, num_classes, train = False)

        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=ssd_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=ssd_dataset_collate)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")


        for param in model.mobilenet.parameters():
            param.requires_grad = True
            
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, criterion, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
