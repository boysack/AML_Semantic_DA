from PIL import Image
import torch.nn as nn
import torch
import numpy as np
import os
import scipy.io as sio

def main():
    """ model1 = CreateModel(args)
    model1.eval()
    model1.cuda()

    model2 = CreateModel(args)
    model2.eval()
    model2.cuda() """


    targetloader = CreateTrgDataLoader(args)

    # ------------------------------------------------- #
    # compute scores and save them
    with torch.no_grad():
        for image, _, name in targetloader:
            # forward
            output1 = model1(image)
            output1 = nn.functional.softmax(output1, dim=1)

            output2 = model2(image)
            output2 = nn.functional.softmax(output2, dim=1)

            a, b = 0.5, 0.5
            output = a*output1 + b*output2 

            output = output.transpose(1,2,0)

            output_nomask = np.asarray( np.argmax(output, axis=2), dtype=np.uint8 )
            output_nomask = Image.fromarray(output_nomask)    
            name = name[0].split('/')[-1]
            output_nomask.save(  '%s/%s' % (args.save, name)  )

    # scores computed and saved
    # ------------------------------------------------- #
    compute_mIoU( args.gt_dir, args.save, args.devkit_dir, args.restore_from )    


if __name__ == '__main__':
    main()