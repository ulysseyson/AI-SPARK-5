""" dir = 'dataset/TRAIN'
import os
# file list
file_list = os.listdir(dir)

local_names = [x.split('.')[0] for x in file_list]

print(local_names) """

local_names = ['공주',
                '노은동',
                '논산',
                '대천2동',
                '독곶리',
                '동문동',
                '모종동',
                '문창동',
                '성성동',
                '신방동',
                '신흥동',
                '아름동',
                '예산군',
                '읍내동',
                '이원면',
                '정림동',
                '홍성읍']