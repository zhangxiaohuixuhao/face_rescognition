import os
import random
from utils import *

DATA_ROOT = r'/home/ma-user/work/face_anti/CVPR19-Face-Anti-spoofing/CASIA-SURF'
#DATA_ROOT = r'/Users/zhaomingming/Documents/CVPRO/face_anti_spoofing_活体检测/CASIA-SURF'
#DATA_ROOT = r'/Users/zhaomingming/Documents/CVPRO/face_anti_spoofing_活体检测/CASIA_SURF'
#DATA_ROOT = r'/Users/zhaomingming/Documents/CVPRO/face_anti_spoofing_活体检测/CASIA_SURF/images/train'
#TRN_IMGS_DIR = DATA_ROOT + '/images/train/Training/'
#TST_IMGS_DIR = DATA_ROOT + '/images/valid/Val/'
DATA_ROOT = r'./'

TRN_IMGS_DIR = DATA_ROOT + './train_data'
RESIZE_SIZE = 112

def load_train_list():
    list = []
    f = open(DATA_ROOT + 'train_data/train_list_small.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_val_list():
    list = []
    f = open(DATA_ROOT + '/val_private_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def load_test_list():
    list = []
    f = open(DATA_ROOT + '/test_public_list.txt')
    lines = f.readlines()

    for line in lines:
        line = line.strip().split(' ')
        list.append(line)

    return list

def transform_balance(train_list):
    print('balance!!!!!!!!')
    pos_list = []
    neg_list = []
    #import pdb
    #pdb.set_trace()
    for tmp in train_list:
        #print(tmp)
        if tmp[3]=='1':
            pos_list.append(tmp)
        else:
            neg_list.append(tmp)

    print("num of pos:%s"%(len(pos_list)))
    print("num of neg:%s"%(len(neg_list)))
    return [pos_list,neg_list]

def submission(probs, outname, mode='valid'):
    if mode == 'valid':
        f = open(DATA_ROOT + '/val_public_list.txt')
    else:
        f = open(DATA_ROOT + '/test_public_list.txt')

    lines = f.readlines()
    f.close()
    lines = [tmp.strip() for tmp in lines]

    f = open(outname,'w')
    for line,prob in zip(lines, probs):
        out = line + ' ' + str(prob)
        f.write(out+'\n')
    f.close()
    return list
