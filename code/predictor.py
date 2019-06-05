import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from utils import Evaluater

import json
from torch import nn
from torchvision import models
from data_loader import TestLoader
from augmentation import OurAug
from LR_predict import LR

class Predictor(Evaluater):
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        self.dataloader.reset()

    def predict(self):
        self.model.eval()
        self.targets, self.scores = np.zeros(self.dataloader.length, dtype=np.int), np.zeros(self.dataloader.length, dtype=np.float64)
        p = 0
        self.loss = 0.
        for i in xrange(int(np.ceil(self.dataloader.length*1.0 / self.dataloader.batch_size))):
            img, target = self.dataloader.generate_batch()
            self.targets[p:p + len(target)] = target
            img, target = Variable(torch.FloatTensor(img).cuda()), Variable(torch.LongTensor(target).cuda())
            score = self.model(img)
            score = F.softmax(score, dim=1).data.cpu().numpy()[:, 1]
            self.scores[p:p + len(score)] = score
            p += len(score)
        return self.scores, self.targets


if __name__ == '__main__':
    config_path = 'test_config.json'
    f = open(config_path, 'r').read()
    config = json.loads(f)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']

    log_path = '%s_%s' %(config['test_lst_path'].split(os.sep)[-1], config['model_path'].split(os.sep)[-2])
    log_path = os.path.join('../prediction', log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path) 
    test_loader = TestLoader(lstpath=config['test_lst_path'], aug=OurAug({'output_shape':config['image_shape']}), batch_size=config['batch_size'], imroot=config['image_root'])
    model = models.resnet50(pretrained=False)
    model.avgpool = nn.AvgPool2d(8)
    model.fc = nn.Linear(2048 * 4, 2)
    model.load_state_dict(torch.load(config['model_path']))
    model.cuda()
    model.eval()

    predictor = Predictor(model, test_loader)
    scores, label_trues = predictor.predict() #quality score predict
    
    #LR eyes model load,predict
    Lr=LR(model_path_LR=config['model_path_LR'], pred_LR_log=config['pred_LR_log'], test_lst_path=config['test_lst_path'])
    names_labels=Lr.LR_predict()
    
    #writing resluts
    with open(os.path.join(log_path, 'name_LR_score_final.txt'), 'w') as f:
        f.write(','.join(['name', 'L/R', 'Score\n']))
        for i, rec in enumerate(test_loader.lst):
            f.write('%s,%s,%s\n' %(names_labels[i][0], names_labels[i][1], scores[i]))
