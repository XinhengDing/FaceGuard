#该文件用于解析数据集

from PIL import Image
from torch.utils.data import Dataset
import os
import random
from random import randint

#用于解析数据集的txt
class MyDataset(Dataset):
    def __init__(self, txt_path_0, txt_path_1, model ,transform=None, target_transform=None):
        fh = open(txt_path_0, 'r')    
        if txt_path_1 is None:
            imgs = []
            for line in fh:
                line = line.rstrip() 
                words = line.split() 
                imgs.append((words[0], int(words[1])))
        else :
            fh1 = open(txt_path_1, "w+")
            fh1.truncate(0)
            if model == 0:
                resultList = random.sample(range(0, 1307167), 388800)
            if model ==1 :
                resultList = random.sample(range(0, 245247), 144000)
            lines = fh.readlines()
            for i in resultList:
                fh1.write(lines[i])
            fh1.close()
            fh2= open(txt_path_1, "r")
            imgs = []
            for line in fh2:
                line = line.rstrip() 
                words = line.split() 
                imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)



