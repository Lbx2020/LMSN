## Lightweight multi-scale network (LMSN) for small object detection
---

## Dataset
1. PASCAL VOC: http://host.robots.ox.ac.uk/pascal/VOC/
2. RSOD: https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-

## Train
1. This article uses the VOC format for training 
   Before training, put the label file in the Annotation under the VOC2007 folder under the VOCdevkit folder.
   Before training, put the image files in JPEGImages under the VOC2007 folder under the VOCdevkit folder.  
2. Use voc_annotation.py to get 2007_train.txt and 2007_val.txt for training.  
3. start network training  

## Predict
1. In the lmsn.py file, modify model_path and classes_path to correspond to the trained files
2. run predict.py  

## Evaluate 
The evaluation results can be obtained by running get_map.py, and the evaluation results will be saved in the map_out folder.

