import os
import cv2
import random
import numpy as np

class DataLoader(object):
    def __init__(self, lstpath, aug, batch_size, imroot):
        self.pointer = 0
        self.lstpath = lstpath
        self.aug = aug
        self.batch_size = batch_size
        self.imroot = imroot

        self.read_list()
        self.reset()

    def read_list(self):
        with open(self.lstpath) as fin:
            self.lst = fin.readlines()
            for i, line in enumerate(self.lst):
                line = line.strip('\r\n').split('/')[-1]
                line1=[line,0]
                self.lst[i] = line1
        self.length = len(self.lst)

    def reset(self):
        pass

    def load_data(self, rec):
        img_name, label = rec
        # print("self.imroot, img_name",self.imroot, img_name)

        if os.path.exists(os.path.join(self.imroot, img_name)):
            img_path = os.path.join(self.imroot, img_name)

        # if os.path.exists(os.path.join(self.imroot, img_name+'.jpeg')):
        #     img_path = os.path.join(self.imroot, img_name+'.jpeg')
        # else:
        #     img_path = os.path.join(self.imroot, img_name+'.jpg')
        # print("img_path",img_path)
        
        assert os.path.exists(img_path)
        # if not os.path.exists(img_path):
        #     print("img_path",img_path)
            
        # assert os.path.exists(img_path)

        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
        img = self.aug.process(img)
        img = np.multiply(img, 1 / 255.0)
        img = np.transpose(img, (2, 0, 1))
        label = int(label)
        return img, label

class TrainLoader(DataLoader):
    def __init__(self, lstpath, aug, batch_size, imroot):
        super(TrainLoader, self).__init__(lstpath, aug, batch_size, imroot)
    
    def generate_batch(self):
        if self.pointer + self.batch_size >= self.length:
            self.reset()
        imgs, labels = [], []
        for i in xrange(self.batch_size):
            rec = self.lst[self.pointer + i]
            im, im_label = self.load_data(rec)
            imgs.append(im)
            labels.append(im_label)

        imgs, labels = np.array(imgs), np.array(labels)

        self.pointer += self.batch_size

        return imgs, labels
    def reset(self):
        self.pointer = 0
        random.shuffle(self.lst)        

class TestLoader(DataLoader):
    def __init__(self, lstpath, aug, batch_size, imroot):
        super(TestLoader, self).__init__(lstpath, aug, batch_size, imroot)
    
    def generate_batch(self):
        imgs, labels = [], []
        for i in xrange(min(self.batch_size, self.length - self.pointer)):
            rec = self.lst[self.pointer + i]
            im, im_label = self.load_data(rec)
            imgs.append(im)
            labels.append(im_label)

        imgs, labels = np.array(imgs), np.array(labels)

        self.pointer += self.batch_size
        if self.pointer >= self.length:
            self.reset()
        return imgs, labels
    def reset(self):
        self.pointer = 0
