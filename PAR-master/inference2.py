import os
import json
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
from net import get_model

import numpy as np


######################################################################
# Settings
# ---------
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }

transforms = T.Compose([
    T.Resize(size=(288, 144)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser()
parser.add_argument('image_path', help='Path to test image')
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--backbone', default='resnet50', type=str, help='model')
parser.add_argument('--use-id', action='store_true', help='use identity loss')
args = parser.parse_args()

assert args.dataset in ['market', 'duke']
assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

model_name = '{}_nfc_id'.format(args.backbone) if args.use_id else '{}_nfc'.format(args.backbone)
# num_label, num_id = num_cls_dict[args.dataset], num_ids_dict[args.dataset]
market_label, market_id = num_cls_dict['market'], num_ids_dict['market']
duke_label, duke_id = num_cls_dict['duke'], num_ids_dict['duke']


######################################################################
# Model and Data
# ---------
def load_network(network, dataset_name):
    save_path = os.path.join('./checkpoints', dataset_name, model_name, 'net_last.pth')
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network

def load_image(path):
    src = Image.open(path)
    print("SRC type {} --> {}".format(type(src),src))
    src = transforms(src)
    print("Post-Transform SRC type {} --> {}".format(type(src),src))
    src = src.unsqueeze(dim=0)
    print("Post unsqueeze SRC type {} --> {}".format(type(src),src))
    return src


# model = get_model(model_name, num_label, use_id=args.use_id, num_id=num_id)
# model = load_network(model)
# model.eval()
model_market = get_model(model_name, market_label, use_id=args.use_id, num_id=market_id)
model_duke = get_model(model_name, duke_label, use_id=args.use_id, num_id=duke_id)
model_market, model_duke = load_network(model_market,'market'), load_network(model_duke, 'duke')
model_market.eval()
model_duke.eval()


src = load_image(args.image_path)

######################################################################
# Inference
# ---------
class predict_decoder(object):

    def __init__(self, dataset):
        with open('./doc/label.json', 'r') as f:
            self.label_list = json.load(f)[dataset]
        with open('./doc/attribute.json', 'r') as f:
            self.attribute_dict = json.load(f)[dataset]
        self.dataset = dataset
        self.num_label = len(self.label_list)

    def decode(self,num_pred, pred):
        print("label_list=",self.label_list)
        # pred = pred.squeeze(dim=0)
        for idx in range(self.num_label):
            name, chooce = self.attribute_dict[self.label_list[idx]]
            # print("idx={} -> {} --> name={}, chooce={}, pred[{}]={}, chooce[pred[idx]]={}"
            # .format(idx,self.label_list[idx],name,chooce,idx,pred[idx],chooce[pred[idx]]))
            if chooce[int(pred[idx])]:
                # print("chooce[pred[idx]]=",chooce[pred[idx]])
                print('{}: {} ({})'.format(name, chooce[int(pred[idx])], round(num_pred[idx],3)))


if not args.use_id:
    out_market = model_market.forward(src)
    out_duke = model_duke.forward(src)
else:
    out_market,_ = model_market.forward(src)
    out_duke,_ = model_duke.forward(src)
print("Pre-pred --> market={}, duke={}".format(out_market, out_duke))

def _combine_models(model1, model2):
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

out = _combine_models(out_market, out_duke)
print("Combined_pred=", out)

pred = np.where(out > 0.5, True, False) #[True if prediction > 0.5 else False for prediction in out]  # threshold=0.5
print("Post-pred =", pred)


Dec = predict_decoder('combined')
Dec.decode(out, pred)
