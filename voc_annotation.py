import os
import random
import xml.etree.ElementTree as ET

from utils.utils import get_classes

#--------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode is used to specify what is calculated when the file is run
#   annotation_mode=0 represents the entire label processing process,
#   annotation_mode=1 means to get the txt in VOCdevkit/VOC2007/ImageSets
#   annotation_mode=2 means to obtain 2007_train.txt, 2007_val.txt for training
#--------------------------------------------------------------------------------------------------------------------------------#
annotation_mode     = 2

classes_path        = 'model_data/RSOD_classes.txt'
#--------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent is used to specify the ratio of (training set + validation set) to test set
#   train_percent is used to specify the ratio of training set to validation set in (training set + validation set)
#--------------------------------------------------------------------------------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9
#-------------------------------------------------------#
#   point to the folder where the dataset is located
#-------------------------------------------------------#
VOCdevkit_path  = 'VOCdevkit'

VOCdevkit_sets  = [('2015', 'train'), ('2015', 'val')]
classes, _      = get_classes(classes_path)

def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'RSOD%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
if __name__ == "__main__":
    random.seed(0)
    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath     = os.path.join(VOCdevkit_path, 'RSOD2015/Annotations')
        saveBasePath    = os.path.join(VOCdevkit_path, 'RSOD2015/ImageSets/Main')
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num     = len(total_xml)  
        list    = range(num)  
        tv      = int(num*trainval_percent)  
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("train size",tr)
        ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)  
        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate train.txt and val.txt for train.")
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'RSOD%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/RSOD%s/JPEGImages/%s.jpg'%(os.path.abspath(VOCdevkit_path), year, image_id))

                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            list_file.close()
        print("Generate train.txt and val.txt for train done.")
