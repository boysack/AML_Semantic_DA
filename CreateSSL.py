from PIL import Image
import torch.nn as nn
import torch
import numpy as np
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
import argparse
from cityscapes import CityScapes
from torch.utils.data import DataLoader
from model.model_stages import BiSeNet
from tqdm import tqdm

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--num_classes',
                       type=int,
                       default=19,
                       help='num of object classes (with void)')
    parse.add_argument('--num_workers',
                       type=int,
                       default=2,
                       help='num of workers')
    parse.add_argument('--use_conv_last',
                       dest='use_conv_last',
                       type=str2bool,
                       default=False,
    )
    parse.add_argument('--backbone',
                       dest='backbone',
                       type=str,
                       default='CatmodelSmall',
    )

    return parse.parse_args()

def main():
    args = parse_args()

    model1 = BiSeNet(backbone=args.backbone, n_classes=args.num_classes, use_conv_last=args.use_conv_last)
    model1.load_state_dict(torch.load('./results-trialFDA-beta0.01/best_FDA_0.01.pth'))
    model2 = BiSeNet(backbone=args.backbone, n_classes=args.num_classes, use_conv_last=args.use_conv_last)
    model2.load_state_dict(torch.load('./results-trialFDA-beta0.05-fromepoch16/best_FDA_0.05.pth'))

    model1.eval()
    model1.cuda()
    model2.eval()
    model2.cuda()

    val_dataset = CityScapes(mode='train', crop=False)
    targetloader = DataLoader(val_dataset,
                       batch_size=1,
                       shuffle=False,
                       num_workers=args.num_workers,
                       drop_last=False)

    #targetloader = CreateTrgDataSSLLoader(args)

    # change the mean for different dataset

    predicted_label = np.zeros((len(targetloader), 512, 1024))
    predicted_prob = np.zeros((len(targetloader), 512, 1024))
    image_name = []

    with torch.no_grad():
        for image, _, name in tqdm(targetloader):
            image = image.cuda()

            # forward
            output1, _, _ = model1(image)
            output1 = nn.functional.softmax(output1, dim=1)

            output2, _, _ = model2(image)
            output2 = nn.functional.softmax(output2, dim=1)


            a, b = 0.5
            output = a*output1 + b*output2 

            output = output.squeeze(0)
            output = output.transpose(1,2,0)
       
            label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
            predicted_label[index] = label.copy()
            predicted_prob[index] = prob.copy()
            image_name.append(name[0])
        
    thres = []
    for i in range(19):
        x = predicted_prob[predicted_label==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x)*0.66))])
    print( thres )
    thres = np.array(thres)
    thres[thres>0.9]=0.9
    print( thres )

    for index in range(len(targetloader)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(19):
            label[   (prob<thres[i]) * (label==i)   ] = 255  
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        output.save(f"./Datasets/Cityscapes/Cityspaces/pseudolabels/{name}.png") 
    
    
if __name__ == '__main__':
    main()
    