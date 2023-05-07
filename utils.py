dir = 'dataset/TRAIN'
import os
# file list
file_list = os.listdir(dir)

local_names = [x.split('.')[0] for x in file_list]

print(local_names)