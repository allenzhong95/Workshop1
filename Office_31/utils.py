import torch.optim as opt
import torch
import numpy as np

from Office_31.basenet import *


def get_model(net, num_class=11, unit_size=100, init=0):
    if net == 'alex':
        model_g = AlexBase()
        model_c = Classifier(num_classes=num_class)
    elif net == 'vgg':
        model_g = VGGBase()
        model_c = Classifier(num_classes=num_class)
    elif net == 'vgg2':
        model_g = VGGBase2()
        model_c = Classifier(num_classes=num_class)
    elif net == 'shallow':
        model_g = Shallow()
        model_c = ClassifierShadow(num_classes=num_class)
    elif net == 'shallow2':
        model_g = Shallow2()
        model_c = Classifier(num_classes=num_class)
    elif net == 'ShallowHome':
        model_g = ShallowHome(init)
        model_c = ClassifierHome(num_classes=num_class, init=init)
    elif net == 'ShallowPie':
        model_g = ShallowPie(init)
        model_c = ClassifierPie(num_classes=num_class)
    elif net == 'ShallowCLEF':
        model_g = ShallowCLEF(init)
        model_c = ClassifierCLEF(num_classes=num_class)
    else:
        model_g = ResBase(net, unit_size=unit_size)
        model_c = ResClassifier(num_classes=num_class, unit_size=unit_size)
    return model_g, model_c


def get_optimizer_visda(lr, G, C, update_lower=False, weight_decay=0.0005):
    if not update_lower:
        params = list(list(G.linear1.parameters()) + list(G.linear2.parameters()) + list(
            G.bn1.parameters()) + list(G.bn2.parameters())) #+ list(G.bn3.parameters()) \
                 #+ list(G.linear3.parameters()) + list(G.linear4.parameters()) + list(G.bn4.parameters())
    else:
        params = G.parameters()
    # optimizer_g = opt.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005,nesterov=True)
    optimizer_g = opt.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay,nesterov=True)
    optimizer_c = opt.SGD(list(C.parameters()), momentum=0.9, lr=lr,
                          weight_decay=weight_decay, nesterov=True)
    return optimizer_g, optimizer_c


def get_optimizer_adm(lr, G, C, update_lower=False, weight_decay=0.0005):
    if not update_lower:
        params = list(list(G.linear1.parameters()) + list(G.linear2.parameters()) + list(
            G.bn1.parameters()) + list(G.bn2.parameters())) #+ list(G.bn3.parameters()) \
                 #+ list(G.linear3.parameters()) + list(G.linear4.parameters()) + list(G.bn4.parameters())
    else:
        params = G.parameters()
    # optimizer_g = opt.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005,nesterov=True)
    optimizer_g = opt.Adam(params, lr=lr, weight_decay=weight_decay)
    optimizer_c = opt.Adam(list(C.parameters()), lr=lr, weight_decay=weight_decay)
    # optimizer_c = opt.SGD(list(C.parameters()), momentum=0.9, lr=lr,
    #                       weight_decay=weight_decay, nesterov=True)
    return optimizer_g, optimizer_c


def get_optimizer_d(lr, model_d, weight_decay=0.005):
    optimizer_d = opt.SGD(list(model_d.parameters()), momentum=0.9, lr=lr,
                          weight_decay=weight_decay, nesterov=True)
    return optimizer_d


def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)


def save_model(model_g, model_c, save_path):
    save_dic = {
        'g_state_dict': model_g.state_dict(),
        'c_state_dict': model_c.state_dict(),
    }
    torch.save(save_dic, save_path)


def load_model(model_g, model_c, load_path):
    checkpoint = torch.load(load_path)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_c.load_state_dict(checkpoint['c_state_dict'])
    return model_g, model_c


def adjust_learning_rate(optimizer, lr, batch_id, max_id, epoch, max_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    beta = 0.75
    alpha = 15
    # alpha = 10  original value
    p = min(1, (batch_id + max_id * epoch) / float(max_id * max_epoch))
    lr = lr / (1 + alpha * p) ** (beta)  # min(1, 2 - epoch/float(20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_learning_rate2(optimizer, lr, lr_rampdown_epochs, epoch, step_in_epoch, total_steps_in_epoch, lr_decay=1):
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    lr *= cosine_rampdown(epoch, lr_rampdown_epochs, decay=lr_decay)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def cosine_rampdown(current, rampdown_length, decay=1):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * (current / rampdown_length)**decay) + 1))
    # return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    cu = np.arange(1, 400)
    for i in cu:
        print(i, cosine_rampdown(i, len(cu)))
