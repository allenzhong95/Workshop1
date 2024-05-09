import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio

import Office_31.utils as utils
import Office_31.loss_cdan as cdan


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def extract_fea(G, C, dataset_test, file_, name='fts'):
    fts = []
    lb = []
    G.eval()
    C.eval()
    for batch_idx, data in enumerate(dataset_test):
        img_t = data[0].float()
        label_t = data[1]
        img_t, label_t = img_t.cuda(), label_t.detach().numpy()
        feat_t = C(G(img_t)).detach().cpu().numpy()
        for i in range(len(feat_t)):
            fts.append(feat_t[i])
            lb.append(label_t[i])
    if not os.path.exists(file_):
        os.makedirs(file_)
    sio.savemat(file_+name+'.mat', {'fts': fts, 'lb': lb})


def pre_train(G, C, opt_g, opt_c, dataset_train, args, eps):
    criterion = nn.CrossEntropyLoss().cuda()
    for ep in range(eps):
        G.train()
        C.train()
        for batch_idx, data in enumerate(dataset_train):
            img_s = data[0].float()
            label_s = data[1]
            img_s, label_s = img_s.cuda(), label_s.cuda()
            if len(img_s) < args.batch_size:
                break
            opt_g.zero_grad()
            opt_c.zero_grad()
            feat_s = G(img_s)
            out_s = C(feat_s)
            loss_s = criterion(out_s, label_s)
            loss_s.backward()

            opt_g.step()
            opt_c.step()
            opt_g.zero_grad()
            opt_c.zero_grad()
        print('pretrain:', ep)


def train(num_epoch, G, C, D, opt_g, opt_c, opt_d, random_layer, dataset_train, dataset_test, result_rc, best_re, best_re2, loss_record, num_class, args):
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_bce = nn.BCELoss().cuda()
    mse = nn.MSELoss().cuda()

    if len(dataset_train) < len(dataset_test):
        pre_train(G, C, opt_g, opt_c, dataset_train, args, 20)
    print('train start!')
    for ep in range(num_epoch):
        G.train()
        C.train()
        D.train()
        loss_epoch = [[], [], []]  # loss_s, loss_t, loos_od
        for batch_idx, data in enumerate(zip(dataset_train, dataset_test)):
            utils.adjust_learning_rate2(opt_g, args.lr, args.epochs+1, ep, batch_idx, len(dataset_train), lr_decay=args.lr_decay)
            utils.adjust_learning_rate2(opt_c, args.lr, args.epochs+1, ep, batch_idx, len(dataset_train), lr_decay=args.lr_decay)
            utils.adjust_learning_rate2(opt_d, args.lr, args.epochs+1, ep, batch_idx, len(dataset_train), lr_decay=args.lr_decay)

            img_s = data[0][0].float()
            label_s = data[0][1]
            img_t = data[1][0].float()
            img_s, label_s = img_s.cuda(), label_s.cuda()
            img_t = img_t.cuda()
            if len(img_t) < args.batch_size:
                break
            if len(img_s) < args.batch_size:
                break
            opt_g.zero_grad()
            opt_c.zero_grad()
            opt_d.zero_grad()
            feat = G(img_s, ep=ep)
            out_s = C(feat)
            loss_s = criterion(out_s, label_s)

            feat_t = G(img_t, ep=ep)
            out_t = C(feat_t)
            out_s_soft = F.softmax(out_s)
            out_t_soft = F.softmax(out_t).detach()

            pre_s = F.softmax(out_s, dim=1)[:, -1]
            pre_t = F.softmax(out_t, dim=1)[:, -1]
            loss_ods = mse(pre_s, torch.from_numpy(np.ones(len(out_s))).cuda().float())
            loss_odt = mse(pre_t, torch.from_numpy(np.ones(len(out_t))).cuda().float())
            if args.relu == 0:
                loss_od = args.alpha * loss_odt - args.gamma * loss_ods
            else:
                # loss_od = args.alpha * loss_odt - args.beta * loss_ods
                # if loss_od < torch.tensor(-0.1):
                #     loss_od = -1 * loss_od
                loss_od = F.relu(args.alpha * loss_odt - args.gamma * loss_ods + args.beta)
                # loss_od = F.relu(args.alpha * loss_odt - args.beta * loss_ods + torch.tensor(0.02))
            print('loss_od', loss_od)

            p = 1.0
            C.set_lambda(p)
            out_t_adv = C(feat_t, reverse=True)
            out_t_adv_prob = F.softmax(out_t_adv)[:, -1]
            target_funk = torch.tensor([args.th] * len(img_t)).cuda()
            loss_t = args.weight_od * criterion_bce(out_t_adv_prob, target_funk)

            loss = loss_s + loss_t + loss_od
            if ep > args.epoch_c:
                out_t_sel = out_t_soft.max(1)[0][out_t.max(1)[1] != 10]
                part_feat_t = feat_t[out_t.max(1)[1] != 10][out_t_sel > args.th_cdan, ]
                lb_s = out_s_soft
                lb_t = out_t_soft[out_t.max(1)[1] != 10][out_t_sel > args.th_cdan, ]
                num = [lb_s.size()[0], lb_t.size()[0]]
                feat_st = torch.cat((feat, part_feat_t), dim=0)
                lb_st = torch.cat((lb_s, lb_t), dim=0)
                entropy = cdan.Entropy(lb_st)
                loss_cdan_kn = cdan.CDAN([feat_st, lb_st], D, entropy, 0.5, random_layer, num)
                # loss_cdan_kn = cdan.CDAN([feat_st, lb_st], D, None, None, random_layer, num)

                out_t_sel = out_t_soft.max(1)[0][out_t.max(1)[1] == 10]
                part_feat_t = feat_t[out_t.max(1)[1] == 10][out_t_sel > args.th_cdan, ]
                feat_st = torch.cat((feat, part_feat_t), dim=0)
                lb_st = torch.cat((out_s_soft, out_t_soft[out_t.max(1)[1] == 10][out_t_sel > args.th_cdan, ]), dim=0)
                num = [len(feat), len(part_feat_t)]
                entropy = cdan.Entropy(lb_st)
                loss_cdan_unk = cdan.CDAN([feat_st, lb_st], D, entropy, 0.5, random_layer, num, reverse=False)
                # loss_cdan_unk = cdan.CDAN([feat_st, lb_st], D, None, None, random_layer, num, reverse=False)
                # loss = loss + args.weight_c1*loss_cdan_unk
                loss = loss + args.weight_c1*loss_cdan_unk + args.weight_c2*loss_cdan_kn

            # tgt_dt = feat_t[out_t_soft.max(1)[0] > args.th_cdan]
            # tgt_lb = out_t.max(1)[1][out_t_soft.max(1)[0] > args.th_cdan].long()
            # if tgt_lb.size()[0] > 5:
            #     out_t2 = C(tgt_dt)
            #     loss += criterion(out_t2, tgt_lb) * 0.1
            loss.backward()

            loss_epoch[0].append(loss_s.item())
            loss_epoch[1].append(loss_t.item())
            loss_epoch[2].append(loss_od.item())

            opt_g.step()
            opt_c.step()
            opt_d.step()
            opt_g.zero_grad()
            opt_c.zero_grad()
            opt_d.zero_grad()

            if batch_idx % args.log_interval == 0:
                print('Train Ep: {} \tLoss Source: {:.6f}\t Loss Target: {:.6f}'.format(
                    ep, loss_s.data.item(), loss_t.data.item()))

        test(ep, G, C, dataset_test, result_rc, best_re, best_re2, num_class, args)
        G.train()
        C.train()
        loss_record[0].append(np.array(loss_epoch[0]).mean())
        loss_record[1].append(np.array(loss_epoch[1]).mean())
        loss_record[2].append(np.array(loss_epoch[2]).mean())

        if args.show == 1:
            plt.cla()
            plt.subplot(221)
            plt.title("args.data:"+str(args.data[0])+'_'+str(args.data[1]))
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

        # if args.save:
        #     if not os.path.exists(args.save_path):
        #         os.mkdir(args.save_path)
        #     utils.save_model(G, C, args.save_path + '_' + str(ep))


def test(ep, G, C, dataset_test, result_rc, best_re, best_re2, num_class, args):
    G.eval()
    C.eval()
    correct = 0
    size = 0
    per_class_num = np.zeros(num_class)
    per_class_correct = np.zeros(num_class).astype(np.float32)
    for batch_idx, data in enumerate(dataset_test):
        img_t, label_t = data[0].float(), data[1]
        img_t, label_t = img_t.cuda(), label_t.cuda()
        feat = G(img_t)
        out_t = C(feat)
        pred = out_t.data.max(1)[1]
        k = label_t.data.size()[0]
        correct += pred.eq(label_t.data).cpu().sum()
        pred = pred.cpu().numpy()
        for t in range(num_class):
            t_ind = np.where(label_t.data.cpu().numpy() == t)
            correct_ind = np.where(pred[t_ind[0]] == t)
            per_class_correct[t] += float(len(correct_ind[0]))
            per_class_num[t] += float(len(t_ind[0]))
        size += k
    per_class_acc = per_class_correct / per_class_num
    result_rc['os_'].append(float(per_class_acc.mean()))
    result_rc['os_star'].append(float(per_class_acc[:-1].mean()))
    result_rc['unknown'].append(per_class_acc[-1])
    if result_rc['os_'][-1] + result_rc['unknown'][-1]*0.75 > best_re['os_'] + best_re['unknown']*0.75:
        best_re['epoch'] = ep
        best_re['os_'] = result_rc['os_'][-1]
        best_re['os_star'] = result_rc['os_star'][-1]
        best_re['unknown'] = result_rc['unknown'][-1]
    if result_rc['os_'][-1] > best_re2['os_']:
        if result_rc['os_star'][-1] - result_rc['os_'][-1] < args.best_th:
            best_re2['epoch'] = ep
            best_re2['os_'] = result_rc['os_'][-1]
            best_re2['os_star'] = result_rc['os_star'][-1]
            best_re2['unknown'] = result_rc['unknown'][-1]
    print('\nTest set including unknown classes:  Accuracy: {}/{} ({:.0f}%)({:.4f}%){}\n'.format(
        correct, size, 100. * correct / size, result_rc['os_'][-1], result_rc['os_star'][-1]))
    print("os:{:.4f}, os*:{:.4f}, unkown:{:.4f}".format(result_rc['os_'][-1], result_rc['os_star'][-1], per_class_acc[-1]))
    print("best: os:{:.4f}, os*:{:.4f}".format(best_re['os_'], best_re['os_star']))
    print("best: os:{:.4f}, os*:{:.4f}".format(best_re2['os_'], best_re2['os_star']))
