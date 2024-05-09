import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
#import models.basenet2 as bakbone2

import torch
import random


def wt_args(args):
    txt = ''
    for ii in args.__dict__.keys():
        txt += str(ii)+':'
        txt += str(args.__dict__[ii])+'\n'
    return txt


def wt_result(file, arr1, arr2, arr3=None):
    f = open(file, 'w+', encoding='utf-8')
    if arr3:
        for ii in range(len(arr1)):
            txt = str(arr1[ii]) + ' ' + str(arr2[ii]) + ' ' + str(arr3[ii]) + '\n'
            f.write(txt)
    else:
        for ii in range(len(arr1)):
            txt = str(arr1[ii]) + ' ' + str(arr2[ii]) + '\n'
            f.write(txt)
    f.close()


def show_training(f1, pause=False):
    f = open(f1, 'r+').read().split('\n')
    while f[-1] == '':
        del f[-1]
    arr1, arr2, arr3 = [], [], []
    for i in f:
        tmp = i.split(' ')
        arr1.append(float(tmp[0]))
        arr2.append(float(tmp[1]))
        arr3.append(float(tmp[2]))
    x = np.arange(0, len(arr1))
    plt.figure()
    if pause:
        plt.ion()
        for i in range(len(arr1)):
            plt.cla()
            plt.plot(x[0:i], arr1[0:i], color='r')
            plt.plot(x[0:i], arr2[0:i], color='g')
            plt.plot(x[0:i], arr3[0:i], color='g')
            plt.pause(0.1)
        plt.ioff()
    plt.plot(x, arr1, color='r')
    plt.plot(x, arr2, color='g')
    plt.plot(x, arr3, color='g')


def show_traing2(result_rc, loss_record, args):
    plt.cla()
    plt.title("args.data:" + str(args.data[0]) + '_' + str(args.data[1]))
    plt.subplot(221)
    plt.xlim(0, len(result_rc['os_']) + 1)
    plt.ylim(0, 1)
    plt.plot(np.arange(0, len(result_rc['os_'])), result_rc['os_'], color='r')
    plt.plot(np.arange(0, len(result_rc['os_star'])), result_rc['os_star'], color='g')
    plt.plot(np.arange(0, len(result_rc['unknown'])), result_rc['unknown'], color='b')
    plt.subplot(222)
    plt.plot(np.arange(0, len(loss_record[0])), loss_record[0], color='r')
    plt.subplot(223)
    plt.plot(np.arange(0, len(loss_record[1])), loss_record[1], color='g')
    plt.subplot(224)
    plt.plot(np.arange(0, len(loss_record[2])), loss_record[2], color='b')
    plt.pause(0.01)


def show_training3(f1, f2):
    f = open(f1, 'r+').read().split('\n')
    while f[-1] == '':
        del f[-1]
    f2 = open(f2, 'r+').read().split('\n')
    while f2[-1] == '':
        del f2[-1]
    arr1, arr2, arr3, arr4, arr5, arr6 = [], [], [], [], [], []
    for i in f:
        tmp = i.split(' ')
        arr1.append(float(tmp[0]))
        arr2.append(float(tmp[1]))
        arr3.append(float(tmp[2]))
    for i in f2:
        tmp = i.split(' ')
        arr4.append(float(tmp[0]))
        arr5.append(float(tmp[1]))
        arr6.append(float(tmp[2]))
    x = np.arange(0, len(arr1))
    plt.figure()
    plt.ylim(0, 1)
    plt.xlim(0, len(x)+1)
    plt.plot(x, arr1, color='r', linestyle="-", marker='.', markersize=1)
    plt.plot(x, arr2, color='g', linestyle="-", marker='.', markersize=1)
    plt.plot(x, arr3, color='b', linestyle="-", marker='*', markersize=1)
    plt.plot(x, arr4, color='r', linestyle="--", marker='*', markersize=1)
    plt.plot(x, arr5, color='g', linestyle="--", marker='*', markersize=1)
    plt.plot(x, arr6, color='b', linestyle="--", marker='*', markersize=1)


def cal_mean(dir_):
    f = open(dir_, 'r+').read().split('\n')
    while f[-1] == '':
        del f[-1]
    arr_tmp = [[], []]
    for ii in f:
        txt = ii.split(' ')
        arr_tmp[0].append(float(txt[0]))
        arr_tmp[1].append(float(txt[1]))
    arr_tmp = np.array(arr_tmp)
    print(np.mean(arr_tmp[0]), np.mean(arr_tmp[1]))


def find_loc(dir_, loc=200, avg_num=1):
    f = open(dir_, 'r+').read().split('\n')
    while f[-1] == '':
        del f[-1]
    os_, os_star, unk = [], [], []
    for i in range(len(f)):
        tmp = f[i].split(' ')
        os_.append(float(tmp[0]))
        os_star.append(float(tmp[1]))
        unk.append(float(tmp[2]))
    re_os = np.mean(os_[loc-avg_num:loc])
    re_os_star = np.mean(os_star[loc-avg_num:loc])
    re_unk = np.mean(unk[loc-avg_num:loc])
    return re_os, re_os_star, re_unk


def find_loc_all(dir_, cycle=5, loc=200, avg_num=1, ed='\n', show_unk=True):
    re_all = []
    all_dir = sorted(os.listdir(dir_))
    # for ii in range(6):
    for ii in range(len(all_dir)):
        re = []
        for jj in range(cycle):
            file = dir_+all_dir[ii]+'/'+str(jj)+'th_result.txt'
            re.append(find_loc(file, loc=loc, avg_num=avg_num))
        re = np.array(re).mean(0)
        re_all.append(re)
    avg = np.array(re_all).mean(0)
    for item in re_all:
        # print(item)
        if show_unk:
            for item_i in item:
                print('%.1f'%(item_i*100), end='')
            print('', end=ed)
        else:
            for item_i in item[0:2]:
                print('&%.1f'%(item_i*100), end='')
            print('', end=ed)
    if show_unk:
        print('avg%.1f%.1f%.1f'%(avg[0]*100, avg[1]*100, avg[2]*100))
    else:
        print('avg&%.1f&%.1f'%(avg[0]*100, avg[1]*100))
    return re_all, avg


def find_best(dir_, th=0.012):
    f = open(dir_, 'r+').read().split('\n')
    while f[-1] == '':
        del f[-1]
    best = [0, 0]
    ind = 0
    # for i in range(1000):
    for i in range(len(f)):
        tmp = f[i].split(' ')
        os_, os_star = float(tmp[0]), float(tmp[1])
        if abs(os_ - os_star) < th:
            if os_ > best[0]:
                best = [os_, os_star]
                ind = i
    print(ind, best)
    return best


def find_best_all(dir_):
    re_all = []
    # th = [0.01, 0.02, 0.02, 0.01, 0.02, 0.02,
    #       0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
    #       0.02, 0.02, 0.02, 0.01, 0.02, 0.01,
    #       0.01, 0.02]
    # th = [0.01, 0.01, 0.015, 0.015, 0.015, 0.015]
    # th = [0.015, 0.015, 0.015, 0.015, 0.015, 0.015,
    #       0.015, 0.015, 0.015, 0.015, 0.015, 0.015]
    all_dir = sorted(os.listdir(dir_))
    # for ii in range(16):
    for ii in range(len(all_dir)):
        re = []
        # for jj in range(int((len(os.listdir(dir_+all_dir[ii]+'/'))-1)/2)):
        for jj in range(5):
            file = dir_+all_dir[ii]+'/'+str(jj)+'th_result.txt'
            re.append(find_best(file, 0.025))
            # re.append(find_best(file, th[ii]))
        re = np.array(re).mean(0)
        re_all.append(re)
    avg = np.array(re_all).mean(0)
    for item in re_all:
        print(item)
    print('avg', avg)
    return re_all, avg


def find_loss_point(dir_):
    f = open(dir_, 'r').read().split('\n')
    while f[-1] == '':
        del f[-1]
    arr = []
    for i in f:
        arr.append(float(i.split(' ')[1]))
    arr = np.array(arr)
    dis = []
    for i in range(200, len(arr)):
        dis.append(np.array(arr[-20+i:-10+i]).mean()-np.array(arr[-10+i:+i]).mean())
        print(i, dis[-1])
        if dis[-1] <= 0:
            print('Stop Flag')
    plt.plot(np.arange(200, len(arr)), dis)


if __name__ == '__main__':
    data_dict = {1: ['amazon.mat', 'dslr.mat'], 2: ['amazon.mat', 'webcam.mat'],
                 3: ['dslr.mat', 'amazon.mat'], 4: ['dslr.mat', 'webcam.mat'],
                 5: ['webcam.mat', 'amazon.mat'], 6: ['webcam.mat', 'dslr.mat']}
    '''1: Analysis os ans os* of A in 5 loops'''
    '''2: Analysis os ans os* between A and B in 5 loops'''
    # rt_dir0 = '/home/bzhang3/zhong/OSBP/Office_31/backup_cdan/'
    rt_dir0 = '/home/bzhang3/zhong/OSBP/Office_31/result/'
    rt_dir = '/home/bzhang3/zhong/OSBP/Office_31/result/'
    num = ['0', '1', '2']
    # num = [1, 2, 3, 4, 5, 6]
    data = 3
    th = '0'
    # for data in num:
    # for th in num:
    #     # f1 = rt_dir0+'relu01/'+data_dict[data][0]+'_'+data_dict[data][1]+'/'+th+'th_result.txt'
    #     f2 = rt_dir+'relutmp2/'+data_dict[data][0]+'_'+data_dict[data][1]+'/'+th+'th_result.txt'
    #     # show_training3(f1, f2)
    #     show_training(f2)
    # plt.show()

    '''find result in at fixed eopch'''
    re_dir = '/home/zhenfang/OPENSET01/Office_31/result3/relu_tmp2/'
    # re_dir0 = '/home/zhenfang/OSBP/Office_31/result3/no_relu_wo_max_wo_c/'
    # re_dir1 = '/home/zhenfang/OSBP/Office_31/result3/no_relu_wo_max_wo_c1u/'
    # re_dir2 = '/home/zhenfang/OSBP/Office_31/result3/no_relu_wo_max_wo_c2k/'
    # re_dir3 = '/home/zhenfang/OSBP/Office_31/result3/no_relu_wo_max_w_c/'
    re_dir4 = '/home/zhenfang/OSBP/Office_31/result3/relu_w_o_c1c2/'
    # re_dir5 = '/home/zhenfang/OSBP/Office_31/result3/relu_w_o_c1u/'
    # re_dir6 = '/home/zhenfang/OSBP/Office_31/result3/relu_w_o_c2k/'
    re_dir7 = '/home/zhenfang/OSBP/Office_31/result3/relu_normal_atlas1/'

    find_loc_all(re_dir, cycle=3, loc=200, avg_num=1, ed=' ', show_unk=False)

    '''find result about specific data at fixed eopch'''
    re = []
    data = 5
    # # for jj in ['1.0', '1.05', '1.1', '1.15', '1.2', '1.25', '1.3', '1.35', '1.4', '1.45', '1.5']:
    # for jj in ['1.0', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7',
    #            '1.8', '1.9', '2.0', '2.1', '2.2', '2.3', '2.4', '2.5']:
    #     for i in range(3):
    #         re_dir = '/home/bzhang3/zhong/OSBP/Office_31/result/relualpha'+jj+'/'
    #         # re_dir = '/home/bzhang3/zhong/OSBP/Office_31/result/no_relugamma1.1beta_0.2/'
    #         re_dir += data_dict[data][0]+'_'+data_dict[data][1]+'/'+str(i)+'th_result.txt'
    #         re.append(find_loc(re_dir, loc=200, avg_num=1))
    #         # print(re[-1])
    #     print(np.mean(re, axis=0), '\n')

    # for i in range(3):
    #     # re_dir = '/home/bzhang3/zhong/OSBP/Office_31/result/relu03/'
    #     re_dir = '/home/bzhang3/zhong/OSBP/Office_31/result/relutmp/'
    #     re_dir += data_dict[data][0]+'_'+data_dict[data][1]+'/'+str(i)+'th_result.txt'
    #     re.append(find_loc(re_dir, loc=200, avg_num=1))
    #     print(re[-1])
    # print(np.mean(re, axis=0), '\n')


