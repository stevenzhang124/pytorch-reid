import os
import json
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torchvision import models
from ..classblock import ClassBlock
from PIL import Image
from torchvision import transforms as T
import argparse
import numpy as np
from pathlib import Path

import numba as nb
from .. import Profiler
import logging
LOGGER = logging.getLogger(__name__)

from multiprocessing.pool import ThreadPool




class Backbone_nFC(nn.Module):
    def __init__(self, class_num, model_name='resnet50_nfc'): #'resnet50_nfc'
        super(Backbone_nFC, self).__init__()
        self.model_name = model_name
        self.backbone_name = model_name.split('_')[0]
        self.class_num = class_num

        model_ft = getattr(models, self.backbone_name)(pretrained=True)
        if 'resnet' in self.backbone_name:
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft
            self.num_ftrs = 2048
        elif 'densenet' in self.backbone_name:
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model_ft.fc = nn.Sequential()
            self.features = model_ft.features
            self.num_ftrs = 1024
        else:
            raise NotImplementedError

        for c in range(self.class_num):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.num_ftrs, class_num=1, activ='sigmoid') )
    
def forward(backbone_M, backbone_D, x):
    with Profiler("feat"):
        x = backbone_M.features(x)
    with Profiler("view"):
        x = x.view(x.size(0), -1)
    with Profiler("lab"):
        pred_label_M, pred_label_D = pred_lab(backbone_M, backbone_D, x)
    with Profiler("tor"):
        pred_label_M = torch.cat(pred_label_M, dim=1).detach().numpy()[0]
        pred_label_D = torch.cat(pred_label_D, dim=1).detach().numpy()[0]
    # LOGGER.info(f"{'Average feature time:':<30}{Profiler.get_avg_millis('feat'):>6.3f} ms")
    # LOGGER.info(f"{'Average view time:':<30}{Profiler.get_avg_millis('view'):>6.3f} ms")
    # LOGGER.info(f"{'Average pred_label time:':<30}{Profiler.get_avg_millis('lab'):>6.3f} ms")
    # LOGGER.info(f"{'Average torch time:':<30}{Profiler.get_avg_millis('tor'):>6.3f} ms")
    return pred_label_M, pred_label_D

# @nb.njit(parallel=True, fastmath=True, cache=True, inline='always')
def pred_lab(backbone_M, backbone_D, x):
    pred_label_M = [backbone_M.__getattr__('class_%d' % c)(x) for c in range(backbone_M.class_num)]
    pred_label_D = [backbone_D.__getattr__('class_%d' % c)(x) for c in range(backbone_D.class_num)]
    return pred_label_M, pred_label_D

def get_model(model_name, num_label, use_id=False, num_id=None):
    if not use_id:
        return Backbone_nFC(num_label, model_name)
    else:
        return Backbone_nFC_Id(num_label, num_id, model_name)

class Decoder(object):
    def __init__(self, dataset):
        with open('./utils/fastmot/doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
            with open('./utils/fastmot/doc/attribute.json', 'r') as f:
                self.attribute_dict = json.load(f)[dataset]
                self.dataset = dataset
                self.num_label = len(self.label_list)
    
    def decode(self, num_pred, pred):
        dico = {}
        luggage = []
        attire = []
        up_color = []
        down_color = []
        for idx in range(self.num_label):
            _, choice = self.attribute_dict[self.label_list[idx]]
            dico[idx] = choice
        gender, hair = dico[4][int(pred[4])], dico[5][int(pred[5])]
        age = dico[0][int(pred[0])] or dico[1][int(pred[1])] or dico[2][int(pred[2])] or dico[3][int(pred[3])]
        # Luggage
        for i in range(6,9):
        	if dico[i][int(pred[i])] == "yes":
        		luggage.append(self.label_list[i])
        luggage = self.joinAttr(luggage,', ') if len(luggage) > 0 else "No"
        # Attire
        for i in range(15,self.num_label):
        	if dico[i][int(pred[i])]:
        		if i < 24:
        			up_color.append(dico[i][int(pred[i])])
        		else:
        			down_color.append(dico[i][int(pred[i])])
        up_color, down_color = self.joinAttr(up_color,' or '), self.joinAttr(down_color,' or ')
        if dico[9][int(pred[9])] == "yes": # Hat
        	attire.append(self.label_list[9])
        attire.append(up_color + " " + dico[10][int(pred[10])]) #Upper Body
        attire.append(down_color + " " + dico[12][int(pred[12])] + " " + dico[11][int(pred[11])]) #Lower Body
        if dico[13][int(pred[13])] == "yes": # Shoes
        	attire.append(dico[14][int(pred[14])] + " " + self.label_list[13])
        else:
            attire.append(dico[14][int(pred[14])] + " " + "shoes")
        attire = self.joinAttr(attire,', ')
        
        return gender, age, hair, luggage, attire
    
    def joinAttr(self, attr, seperator):
        return seperator.join(''.join(list(filter(None.__ne__, i))) for i in attr) if len(attr) > 0 else ""

class AttributeRecognizer:
    def __init__(self, backbone='resnet50'):

        self.transforms = T.Compose([
            T.Resize(size=(288, 144)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.backbone = backbone
        assert backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

        num_cls_dict = { 'market':30, 'duke':23 }
        num_ids_dict = { 'market':751, 'duke':702 }
        model_name = '{}_nfc'.format(backbone)
        
        market_label, market_id = num_cls_dict['market'], num_ids_dict['market']
        duke_label, duke_id = num_cls_dict['duke'], num_ids_dict['duke']
        with open('./utils/fastmot/doc/label.json', 'r') as f:
            self.label_list_m = json.load(f)['market']
            with open('./utils/fastmot/doc/attribute.json', 'r') as f:
                self.attribute_dict_m = json.load(f)['market']
        with open('./utils/fastmot/doc/label.json', 'r') as f:
            self.label_list_d = json.load(f)['duke']
            with open('./utils/fastmot/doc/attribute.json', 'r') as f:
                self.attribute_dict_d = json.load(f)['duke']

        self.network_M = get_model(model_name, market_label, use_id=False, num_id=market_id)
        self.network_D = get_model(model_name, duke_label, use_id=False, num_id=duke_id)
        self.load_networks(model_name)
        self.decoder = Decoder('combined')
        # self.pool = ThreadPool()
        
    # def __del__(self):
    #     self.pool.close()
    #     self.pool.join()


    def load_networks(self, model_name):
        save_path_M = os.path.join('./utils/fastmot/checkpoints', 'market', model_name, 'net_last.pth')
        save_path_D = os.path.join('./utils/fastmot/checkpoints', 'duke', model_name, 'net_last.pth')
        self.network_M.load_state_dict(torch.load(save_path_M))
        self.network_D.load_state_dict(torch.load(save_path_D))
        self.network_M.eval()
        self.network_D.eval()
    
    def predict(self,img):
        with Profiler("load"):
            src = self.load_image(img)
        with Profiler("forward"):
            out_market, out_duke = forward(self.network_M, self.network_D, src)
        with Profiler("filter"):
            out_market_2 = np.where(out_market > 0.5, True, False)
            out_duke_2 = np.where(out_duke > 0.5, True, False)
        with Profiler("comb"):
            # print("MARKET ->",out_market)
            # print("DUKE -> ", out_duke)
            out = _combine_outputs(out_market, out_duke)
        # print("Combined_pred=", out)

        # LOGGER.info(f"{'Average image loading time:':<30}{Profiler.get_avg_millis('load'):>6.3f} ms")
        # LOGGER.info(f"{'Average forward time:':<30}{Profiler.get_avg_millis('mf'):>6.3f} ms")
        # LOGGER.info(f"{'Average filter time:':<30}{Profiler.get_avg_millis('filter'):>6.3f} ms")
        # LOGGER.info(f"{'Average combination time:':<30}{Profiler.get_avg_millis('comb'):>6.3f} ms")
        pred = np.where(out > 0.5, True, False) #[True if prediction > 0.5 else False for prediction in out]  # threshold=0.5
        #pred = out #[True if prediction > 0.5 else False for prediction in out]  # threshold=0.5
        #print("Post-pred =", pred)
        return self.decoder.decode(out, pred)
    
    def load_image(self, img):
        # src = Image.open(path)
        src = Image.fromarray(np.uint8(img[:, :, [2, 1, 0]]))
        # src.show()
        src = self.transforms(src)
        src = src.unsqueeze(dim=0)
        return src

    
# @nb.njit(parallel=True, fastmath=True, cache=True, inline='always')
def _combine_outputs(market, duke):
    comb = np.zeros(34) # 15 unique + 19 shared

    # Physical feats
    comb[:4] = market[:4] # Age
    comb[4] = min(market[12],duke[4]) #Gender
    comb[5] = market[10] # Hair
    # Accesories
    for i, j, k in zip(range(6,9), range(4,7), range(3)): # Lugage(backpack, bag, handbag)
        comb[i] = min(market[j], duke[k])
    # Clothing (top --> bottom)
    comb[9] = min(market[11], duke[5]) # Hat
    comb[10] = market[9] # Length upper body
    comb[11], comb[12] = market[7], market[8] # Type, length lower body
    comb[13], comb[14] = duke[3], duke[6] # Boots, Shoes
    # Color
    for i, j, k in zip(range(15,22), [13, 14, 15, 16, 18, 19, 20], range(8,15)): # Upper body --> (black, white, red, purple. grey. blue, green)
        comb[i] = market[j] #max(market[j],duke[k]) # --> (black, white, red, purple. grey. blue, green)
        comb[22] = market[17] # yellow
        comb[23] = duke[15] # brown
    for i, j, k in zip(range(24,30), [21, 22, 26, 27, 28, 29], [16, 17, 19, 20, 21, 22]): # Lower body
        comb[i] = duke[k]# max(market[j], duke[k]) # --> (black, white, grey, blue green, brown)
        for i, j in zip(range(30,33), range(23,26)):
            comb[i] = market[j]                    # --> (pink, purple, yellow)
            comb[33] = duke[18]                      # --> brown

    return comb
