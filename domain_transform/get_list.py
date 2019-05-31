from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import numpy as np
from PIL import Image

path = "/home/yq/Downloads/IJCAI/IJCAI_fgsm_output/"
#path = "/home/yq/Downloads/IJCAI/attack_samples/samples/" 
#path = "/home/yq/Downloads/IJCAI/ATN_train_output/"

train_list = list()
val_list = list()
for first_file in os.listdir(path):
    first_path = os.path.join(path, first_file)
    if os.path.isdir(first_path):
        num = 0
        for second_file in os.listdir(first_path):
            num = num + 1    
            extension = second_file.split('.')[-1]
            if extension == 'jpg':
                fileLoc = os.path.join(first_path,second_file)
                raw_image = Image.open(fileLoc)
                if raw_image.mode == 'RGB'and num < 1000:
                    if random.random() > 0.15:
                        train_list.append(fileLoc)
                    else:
                        val_list.append(fileLoc)

print("train list number: ", len(train_list))
print("val list number: ", len(val_list))

with open("fgsm_train_list.txt","w") as f:
    for name in train_list:
        f.write(name)
        f.write("\n")

with open("fgsm_val_list.txt","w") as f:
    for name in val_list:
        f.write(name)
        f.write("\n")
