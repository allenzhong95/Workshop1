"""
Introduction:
python trainer_osda_fea_01.py --data 3 --alpha 1 --beta 0.5 --batch-size 128 --lr 0.001 --epochs 1000
"""

from __future__ import print_function
import argparse
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import torch
import random

import sys
import Office_31.utils as utils
import Office_31.result_analysis as re_ana
from Office_31.get_office_fea import get_dataset
import Office_31.train_of31_cdan02 as train
import Office_31.basenet as basenet

parser = argparse.ArgumentParser(description='PyTorch Openset DA')
parser.add_argument('--batch_size', type=int, default=128, metavar='N')
parser.add_argument('--epochs', type=int, default=200, metavar='N')
parser.add_argument('--epoch_c', type=int, default=10, metavar='N')
parser.add_argument('--lr', type=float, default=0.0025, metavar='LR')
parser.add_argument('--lr_decay', type=int, default=1, metavar='LR')
parser.add_argument('--data', type=str, default='32', help='1A,2D,3W')
parser.add_argument('--Num_class', type=int, default=11)
parser.add_argument('--gpu_id', type=str, default='0')

parser.add_argument('--weight_od', type=float, default=1, help='default: ')
parser.add_argument('--weight_c1', type=float, default=1, help='default: ')
parser.add_argument('--weight_c2', type=float, default=1, help='default: ')
parser.add_argument('--alpha', type=float, default=1.25)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument('--relu', type=int, default=1, help='default:0')
parser.add_argument('--file', type=str, default='tmp2')
parser.add_argument('--show', type=int, default=0, help='default:0')
parser.add_argument('--weight-decay', type=float, default=1e-3)
parser.add_argument('--best_th', type=float, default=0.01, help='the threshold of best result:')
parser.add_argument('--th', type=float, default=0.5, help='adversarial threshold')
parser.add_argument('--th_cdan', type=float, default=0.9, help='adversarial threshold')
parser.add_argument('--net', type=str, default='shallow', metavar='B')
parser.add_argument('--init', type=int, default=0)

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N')
parser.add_argument('--save', action='store_true', default=True, help='save model or not')
parser.add_argument('--save_path', type=str, default='/home/bzhang3/zhong/OSBP/Office_31/checkpoint/', metavar='B', help='checkpoint path')
parser.add_argument('--result_path', type=str, default='./tnnls_result_of31/')
parser.add_argument('--data_path', type=str, default='D:/download/TNNLS_AAAI/Data/Office31_Vgg19_relu1_have_avgp/')
parser.add_argument('--unit_size', type=int, default=1000, metavar='N', help='unit size of fully connected layer')
parser.add_argument('--update_lower', action='store_true', default=True, help='update lower layer or not')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disable cuda')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
args.cuda = not args.no_cuda and torch.cuda.is_available()


if __name__ == '__main__':
    train.setup_seed(args.seed)
    paras = re_ana.wt_args(args)
    record = [paras]
    best_record_all = {1: [], 2: []}
    root_path = args.data_path
    data_dict = {1: 'amazon.mat', 2: 'dslr.mat', 3: 'webcam.mat'}

    for i in range(3):
        source_name = data_dict[int(args.data[0])]
        target_name = data_dict[int(args.data[1])]
        source_dataset, target_dataset = get_dataset(root_path, source_name, target_name)
        dt_train = torch.utils.data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        dt_test = torch.utils.data.DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        model_G, model_C = utils.get_model(args.net, num_class=args.Num_class, unit_size=args.unit_size, init=args.init)
        print(model_G, '\n', model_C)
        model_G, model_C = model_G.cuda(), model_C.cuda()
        model_D = basenet.AdversarialNetwork(2048 * 11, 500).cuda()
        Opt_d = utils.get_optimizer_d(args.lr, model_D, weight_decay=args.weight_decay)
        Opt_g, Opt_c = utils.get_optimizer_visda(args.lr, model_G, model_C, update_lower=args.update_lower,
                                                 weight_decay=args.weight_decay)
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)
        if args.relu == 1:
            save_dir = args.result_path+'relu_' + args.file + '/' + source_name + '_' + target_name
        else:
            save_dir = args.result_path+'no_relu_' + args.file + '/' + source_name + '_' + target_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print('created new file directory.')

        best_record = {'epoch': 0, 'os_': 0, 'os_star': 0, 'unknown': 0}
        best_record2 = {'epoch': 0, 'os_': 0, 'os_star': 0, 'unknown': 0}
        result_record = {'os_': [], 'os_star': [], 'unknown': []}
        loss_record = [[], [], []]  # loss_s, loss_t, loos_od

        try:
            print(paras)
            train.train(args.epochs + 1, model_G, model_C, model_D, Opt_g, Opt_c, Opt_d, None, dt_train,
                        dt_test, result_record, best_record, best_record2, loss_record, args.Num_class, args)
            # train.extract_fea(model_G, model_C, dt_test, '/home/bzhang3/zhong/OSBP/Office_31/feature2/', 'target'+args.data)
            # train.extract_fea(model_G, model_C, dt_train, '/home/bzhang3/zhong/OSBP/Office_31/feature2/', 'source'+args.data)
            # '''save_model'''
            # if args.save:
            #     if not os.path.exists(args.save_path+args.file):
            #         os.mkdir(args.save_path+args.file)
            #     utils.save_model(model_G, model_C, args.save_path + args.file + '/' + args.data + str(i))
            '''save last records, best records and parameters'''
            best_record_all[1].append([best_record['os_'], best_record['os_star']])
            best_record_all[2].append([best_record2['os_'], best_record2['os_star']])
            last_re = "average on last 5 epochs: os:{}, os*:{}".format(
                np.array(result_record['os_'][-5:]).mean(), np.array(result_record['os_star'][-5:]).mean())
            print(last_re)
            '''save files(result, loss) and image'''
            if args.show == 2:
                re_ana.show_traing2(result_record, loss_record, args)
                # plt.savefig(save_dir+'/'+str(i)+'th.jpg')
                plt.show()
            result_paras = 'best(epoch/os/os/unknown*):' + str(best_record['epoch']) + ' ' + \
                           str(best_record['os_']) + ' ' + str(best_record['os_star']) + ' ' + \
                           str(best_record['unknown']) + '\nbest2(epoch/os/os/unknown*):' + \
                           str(best_record2['epoch']) + ' ' + str(best_record2['os_']) + ' ' + \
                           str(best_record2['os_star']) + ' ' + str(best_record2['unknown']) + \
                           '\n' + last_re + '\n'
            record.append(str(i) + 'th\n' + result_paras)
            re_ana.wt_result(save_dir + '/' + str(i) + 'th_result.txt', result_record['os_'], result_record['os_star'],
                             result_record['unknown'])
            time.sleep(0.5)
            re_ana.wt_result(save_dir + '/' + str(i) + 'th_loss.txt', loss_record[0], loss_record[1], loss_record[2])
            time.sleep(0.5)
            if i == 0:
                f = open(save_dir + '/result_paras.txt', 'w+', encoding='utf-8')
                f.write(paras + '\n' + result_paras)
                f.close()
            else:
                f = open(save_dir + '/result_paras.txt', 'a+', encoding='utf-8')
                f.write(result_paras)
                f.close()
        except KeyboardInterrupt:
            result_paras = 'best(epoch/os/os*/unknown):' + str(best_record['epoch']) + ' ' + \
                           str(best_record['os_']) + ' ' + str(best_record['os_star']) + ' ' + \
                           str(best_record['unknown']) + '\nbest2(epoch/os/os*/unknown):' + \
                           str(best_record2['epoch']) + ' ' + str(best_record2['os_']) + ' ' + \
                           str(best_record2['os_star']) + ' ' + str(best_record2['unknown']) + '\n'
            print('Error!\n', 'best:', result_paras)
            print('best result is: ', np.array(best_record_all[1]).mean(0), '\n', np.array(best_record_all[2]).mean(0))
    for i in record:
        print(i)
    best_txt = 'best result is: best1={},\nbest2={}\n'.format(np.array(best_record_all[1]).mean(0),
                                                              np.array(best_record_all[2]).mean(0))
    print(best_txt)
