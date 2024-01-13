#!/usr/bin/python
# -*- encoding: utf-8 -*-
from model.model_stages import BiSeNet
from cityscapes import CityScapes
import torch
from torch.utils.data import DataLoader
import logging
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.cuda.amp as amp
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from tqdm import tqdm
from model.discriminator import FCDiscriminator
import torch.nn.functional as F
import os
from gta import GTA


logger = logging.getLogger()


def val(args, model, dataloader):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict, _, _ = model(data)
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou


def train(args, model, model_D1, model_D2, model_D3, optimizer, optimizer_D1, optimizer_D2, optimizer_D3, dataloader_train, dataloader_target, dataloader_val):
    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    bce_loss = torch.nn.BCEWithLogitsLoss()

    source_label = 0
    target_label = 1


    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        lr_D1 = poly_lr_scheduler(optimizer_D1, args.learning_rate_D, iter=epoch, max_iter=args.num_epochs)
        lr_D2 = poly_lr_scheduler(optimizer_D2, args.learning_rate_D, iter=epoch, max_iter=args.num_epochs)
        lr_D3 = poly_lr_scheduler(optimizer_D3, args.learning_rate_D, iter=epoch, max_iter=args.num_epochs)

        model.train()
        model_D1.train()
        model_D2.train()
        model_D3.train()

        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []

        for (data, label), (data_t, _) in zip(dataloader_train, dataloader_target):
            torch.cuda.empty_cache()     
            data = data.cuda()
            label = label.long().cuda()
            data_t = data_t.cuda()

            optimizer.zero_grad()
            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            optimizer_D3.zero_grad()

            # train G
            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False
            for param in model_D3.parameters():
                param.requires_grad = False

            with amp.autocast():
                output, out16, out32 = model(data)
                output_t, out16_t, out32_t= model(data_t)

                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

            scaler.scale(loss).backward()

            with amp.autocast():
                D_out1= model_D1(F.softmax(output_t, dim=1))
                D_out2= model_D2(F.softmax(out16_t, dim=1))
                D_out3= model_D3(F.softmax(out32_t, dim=1))

                loss_d_t1= bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda())
                loss_d_t2= bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).cuda())
                loss_d_t3= bce_loss(D_out3, torch.FloatTensor(D_out3.data.size()).fill_(source_label).cuda())

                loss_d_t = args.lambda_d1 * loss_d_t1 + args.lambda_d2 * loss_d_t2 + args.lambda_d3 * loss_d_t3

            scaler.scale(loss_d_t).backward()

            #train D
            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True
            for param in model_D3.parameters():
                param.requires_grad = True
            
            output = output.detach()
            out16 = out16.detach()
            out32 = out32.detach()

            with amp.autocast():
                D_out1 = model_D1(F.softmax(output, dim=1))
                D_out2 = model_D2(F.softmax(out16, dim=1))
                D_out3 = model_D3(F.softmax(out32, dim=1))

                loss_d_s1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).cuda())
                loss_d_s2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).cuda())
                loss_d_s3 = bce_loss(D_out3, torch.FloatTensor(D_out3.data.size()).fill_(source_label).cuda())
                

            scaler.scale(loss_d_s1).backward()
            scaler.scale(loss_d_s2).backward()
            scaler.scale(loss_d_s3).backward()

            output_t = output_t.detach()
            out16_t = out16_t.detach()
            out32_t = out32_t.detach()

            with amp.autocast():
                D_out1 = model_D1(F.softmax(output_t, dim=1))
                D_out2 = model_D2(F.softmax(out16_t, dim=1))
                D_out3 = model_D3(F.softmax(out32_t, dim=1))

                loss_d_t1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).cuda())
                loss_d_t2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).cuda())
                loss_d_t3 = bce_loss(D_out3, torch.FloatTensor(D_out3.data.size()).fill_(target_label).cuda())


            scaler.scale(loss_d_t1).backward()
            scaler.scale(loss_d_t2).backward()
            scaler.scale(loss_d_t3).backward()

            scaler.step(optimizer)
            scaler.step(optimizer_D1)
            scaler.step(optimizer_D2)
            scaler.step(optimizer_D3)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1

            loss_record.append(loss.item())

        tq.close()
        loss_train_mean = np.mean(loss_record)

        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))
            torch.save(model_D1.module.state_dict(), os.path.join(args.save_model_path, 'latest_D1.pth'))
            torch.save(model_D2.module.state_dict(), os.path.join(args.save_model_path, 'latest_D2.pth'))
            torch.save(model_D3.module.state_dict(), os.path.join(args.save_model_path, 'latest_D3.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
                torch.save(model_D1.module.state_dict(), os.path.join(args.save_model_path, 'best_D1.pth'))
                torch.save(model_D2.module.state_dict(), os.path.join(args.save_model_path, 'best_D2.pth'))
                torch.save(model_D3.module.state_dict(), os.path.join(args.save_model_path, 'best_D3.pth'))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()

    parse.add_argument('--mode',
                       dest='mode',
                       type=str,
                       default='train',
    )

    parse.add_argument('--backbone',
                       dest='backbone',
                       type=str,
                       default='CatmodelSmall',
    )
    parse.add_argument('--pretrain_path',
                      dest='pretrain_path',
                      type=str,
                      default='./STDCNet813M_73.91.tar',
    )
    parse.add_argument('--use_conv_last',
                       dest='use_conv_last',
                       type=str2bool,
                       default=False,
    )
    parse.add_argument('--num_epochs',
                       type=int, default=50,
                       help='Number of epochs to train for')
    parse.add_argument('--epoch_start_i',
                       type=int,
                       default=0,
                       help='Start counting epochs from this number')
    parse.add_argument('--checkpoint_step',
                       type=int,
                       default=10,
                       help='How often to save checkpoints (epochs)')
    parse.add_argument('--validation_step',
                       type=int,
                       default=1,
                       help='How often to perform validation (epochs)')
    parse.add_argument('--crop_height',
                       type=int,
                       default=512,
                       help='Height of cropped/resized input image to modelwork')
    parse.add_argument('--crop_width',
                       type=int,
                       default=1024,
                       help='Width of cropped/resized input image to modelwork')
    parse.add_argument('--batch_size',
                       type=int,
                       default=8,
                       help='Number of images in each batch')
    parse.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='learning rate used for train')
    parse.add_argument('--learning_rate_D',
                        type=float,
                        default=0.0002,
                        help='learning rate used for train')
    parse.add_argument('--num_workers',
                       type=int,
                       default=2,
                       help='num of workers')
    parse.add_argument('--num_classes',
                       type=int,
                       default=19,
                       help='num of object classes (with void)')
    parse.add_argument('--cuda',
                       type=str,
                       default='0',
                       help='GPU ids used for training')
    parse.add_argument('--use_gpu',
                       type=bool,
                       default=True,
                       help='whether to user gpu for training')
    parse.add_argument('--save_model_path',
                       type=str,
                       default='./results',
                       help='path to save model')
    parse.add_argument('--optimizer',
                       type=str,
                       default='adam',
                       help='optimizer, support rmsprop, sgd, adam')
    parse.add_argument('--loss',
                       type=str,
                       default='crossentropy',
                       help='loss function')
    parse.add_argument('--lambda_d1',
                       type=float,
                       default=0.001,
                       help='lambda for adversarial loss')
    parse.add_argument('--lambda_d2',
                       type=float,
                       default=0.0002,
                       help='lambda for adversarial loss')
    parse.add_argument('--lambda_d3',
                       type=float,
                       default=0.0002,
                       help='lambda for adversarial loss')
    


    return parse.parse_args()


def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes

    mode = args.mode

    train_dataset = GTA('all')
    dataloader_train = DataLoader(train_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=False,
                    drop_last=True)

    target_dataset = CityScapes(mode, max_iter=len(train_dataset))
    dataloader_target = DataLoader(target_dataset,
                       batch_size=args.batch_size,
                       shuffle=False,
                       num_workers=args.num_workers,
                       drop_last=False)
    
    val_dataset = CityScapes(mode='val')
    dataloader_val = DataLoader(val_dataset,
                       batch_size=1,
                       shuffle=False,
                       num_workers=args.num_workers,
                       drop_last=False)

    ## model
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)
    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D2 = FCDiscriminator(num_classes=args.num_classes)
    model_D3 = FCDiscriminator(num_classes=args.num_classes)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        model_D1 = torch.nn.DataParallel(model_D1).cuda()
        model_D2 = torch.nn.DataParallel(model_D2).cuda()
        model_D3 = torch.nn.DataParallel(model_D3).cuda()

    ## optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None
    
    optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2 = torch.optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D3 = torch.optim.Adam(model_D3.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))

    
    ## train loop
    train(args, model, model_D1, model_D2, model_D3,  optimizer, optimizer_D1, optimizer_D2, optimizer_D3, dataloader_train, dataloader_target, dataloader_val)
    # final test
    val(args, model, dataloader_val)

if __name__ == "__main__":
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "heuristic"
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "pooled"
    main()