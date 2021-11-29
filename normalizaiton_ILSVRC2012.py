# -*- coding: utf-8 -*-
"""
Normalize ILSVCR2012 Database 

Created on Wed Oct 20 14:38:45 2021

@author: yiqi.she
"""

import os
import tarfile

database_path = 'D:/imagenet'
train_dataset_tar = 'ILSVRC2012_img_train.tar'
val_dataset_tar = 'ILSVRC2012_img_val.tar'

## mkdir ILSVRC2012
output_path = os.path.join(database_path, 'ILSVRC2012')    
if os.path.isdir(output_path):
    pass
else:
    os.mkdir(output_path)
    
## uncompress train dataset
train_path_tar = os.path.join(database_path, train_dataset_tar)
train_path = os.path.join(output_path,'train_dataset')
#tar_file = tarfile.open(train_path_tar)
#tar_file.extractall(train_path)
#tar_file.close()

for i_file in (os.listdir(train_path)):
    print (i_file)
    i_tar_file = tarfile.open(os.path.join(train_path, i_file))
    i_tar_file.extractall(os.path.join(train_path, str(i_file).split('.')[0]))
    i_tar_file.close()
    os.remove(os.path.join(train_path, i_file))

## uncompress validation set
val_path_tar = os.path.join(database_path, val_dataset_tar)
val_path = os.path.join(output_path,'validation_dataset')
val_file = tarfile.open(val_path_tar)
val_file.extractall(val_path)
val_file.close()
