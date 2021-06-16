import numpy as np
import sys
import os


def get_file_names(path):

    names = []
    for root,dir,filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.txt'):
                names.append(filename)

    return names



def convert_data(names,img_path,label_path):

    fd = open('train.txt','w')
    fl = open('label.txt','w')

    for name in names:
        img_name = name.split('.')[0]
        img_name = img_path + '/' + img_name + '.jpg' + '\n'
        fd.write(img_name)

        label_name = label_path + '/' + name
        f = open(label_name,'r')
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            for l in line:
                fl.write(l + ' ')
            fl.write('\n')


if __name__ == '__main__':
    img_path = 'J:/dataset/coco2017/train2017/images'
    label_path = 'J:/dataset/coco2017/train2017/labels'
    names = get_file_names(label_path)
    convert_data(names,img_path,label_path)










