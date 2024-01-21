from PIL import Image
import torch.nn as nn
import torch
import numpy as np
import os
import scipy.io as sio

def main():
    """ model1 = CreateSSLModel(args)
    model1.eval()
    model1.cuda()

    args.restore_from = args.restore_opt2
    model2 = CreateSSLModel(args)
    model2.eval()
    model2.cuda()

    args.restore_from = args.restore_opt3
    model3 = CreateSSLModel(args)
    model3.eval()
    model3.cuda() """

    #targetloader = CreateTrgDataSSLLoader(args)

    # change the mean for different dataset

    predicted_label = np.zeros((len(targetloader), 512, 1024))
    predicted_prob = np.zeros((len(targetloader), 512, 1024))
    image_name = []

    with torch.no_grad():
        for index, batch in enumerate(targetloader):
            if index % 100 == 0:
                print( '%d processd' % index )
            image, _, name = batch

            # forward
            output1 = model1(image)
            output1 = nn.functional.softmax(output1, dim=1)

            output2 = model2(image)
            output2 = nn.functional.softmax(output2, dim=1)


            a, b = 0.5
            output = a*output1 + b*output2 

            #output = nn.functional.interpolate(output, (512, 1024), mode='bilinear', align_corners=True).cpu().data[0].numpy()
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
        name = name.split('/')[-1]
        output.save('%s/%s' % (, name)) 
    
    
if __name__ == '__main__':
    main()
    