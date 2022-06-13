import os
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import get_model

import numpy as np

class Decoder(object):
    def __init__(self, dataset):
        with open('./doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
            with open('./doc/attribute.json', 'r') as f:
                self.attribute_dict = json.load(f)[dataset]
                self.dataset = dataset
                self.num_label = len(self.label_list)

    def decode(self, num_pred, pred):
        print("label_list=",self.label_list)
        # pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            # print("idx={} -> {} --> name={}, chooce={}, pred[{}]={}, chooce[pred[idx]]={}"
            # .format(idx,self.label_list[idx],name,chooce,idx,pred[idx],chooce[pred[idx]]))
            if chooce[int(pred[idx])]:
                # print("chooce[pred[idx]]=",chooce[pred[idx]])
                print('{}: {} ({})'.format(name, chooce[int(pred[idx])], round(num_pred[idx],3)))

class AttributeRecognition():
    def __init__(self, backbone='resnet50'):

    self.dataset_dict = {
        'market'  :  'Market-1501',
        'duke'  :  'DukeMTMC-reID',
    }
    self.num_cls_dict = { 'market':30, 'duke':23 }
    self.num_ids_dict = { 'market':751, 'duke':702 }

    self.transforms = T.Compose([
        T.Resize(size=(288, 144)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    self.backbone = backbone
    assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

    self.model_name = '{}_nfc'.format(args.backbone)
    market_label, market_id = num_cls_dict['market'], num_ids_dict['market']
    duke_label, duke_id = num_cls_dict['duke'], num_ids_dict['duke']

    self.network_M = get_model(model_name, market_label, use_id=args.use_id, num_id=market_id)
    self.network_D = get_model(model_name, duke_label, use_id=args.use_id, num_id=duke_id)
    self.load_networks()
    self.decoder = Decoder('combined')

    def load_networks(self, dataset_name):
        save_path_M = os.path.join('./checkpoints', 'market', model_name, 'net_last.pth')
        save_path_D = os.path.join('./checkpoints', 'duke', model_name, 'net_last.pth')
        self.network_M.load_state_dict(torch.load(save_path_M))
        self.network_D.load_state_dict(torch.load(save_path_D))
        self.network_M.eval()
        self.network_D.eval()
        print('Resume model from {}'.format(save_path_M))
        print('Resume model from {}'.format(save_path_D))


    def predict(self,img):
        src = load_image(img)
        out_market = self.network_M.forward(src)
        out_duke = self.network_D.forward(src)
        print("Pre-pred --> market={}, duke={}".format(out_market, out_duke))

        out = _combine_outputs(out_market, out_duke)
        print("Combined_pred=", out)

        pred = np.where(out > 0.5, True, False) #[True if prediction > 0.5 else False for prediction in out]  # threshold=0.5
        print("Post-pred =", pred)

        self.decoder.decode(out, pred)

    def load_image(img):
        # src = Image.open(path)
        src = transforms(img)
        src = src.unsqueeze(dim=0)
        return src

    def _combine_outputs(model1, model2):
        comb = np.zeros(34) # 15 unique + 19 shared

        # Physical feats
        comb[:4] = model1[:4] # Age
        comb[4] = min(model1[12],model2[4]) #Gender
        comb[5] = model1[10] # Hair
        # Accesories
        for i, j, k in zip(range(6,9), range(4,7), range(3)): # Lugage(backpack, bag, handbag)
        comb[i] = min(model1[j], model2[k])
        # Clothing (top --> bottom)
        comb[9] = min(model1[11], model2[5]) # Hat
        comb[10] = min(model1[9], model2[7]) # Length upper body
        comb[11], comb[12] = model1[7], model1[8] # Type, length lower body
        comb[13], comb[14] = model2[3], model2[6] # Boots, Shoes
        # Color
        for i, j, k in zip(range(15,22), [13, 14, 15, 16, 18, 19, 20], range(8,15)): # Upper body --> (black, white, red, purple. grey. blue, green)
        comb[i] = max(model1[j], model2[k]) # --> (black, white, red, purple. grey. blue, green)
        comb[22] = model1[17] # yellow
        comb[23] = model2[15] # brown
        for i, j, k in zip(range(24,30), [21, 22, 26, 27, 28, 29], [16, 17, 19, 20, 21, 22]): # Lower body
        comb[i] = max(model1[j], model2[k]) # --> (black, white, grey, blue green, brown)
        for i, j in zip(range(30,33), range(23,26)):
        comb[i] = model1[j]                    # --> (pink, purple, yellow)
        comb[33] = model2[18]                      # --> brown

        return comb
