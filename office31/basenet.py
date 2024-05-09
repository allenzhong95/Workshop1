from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
import torch
import math
import numpy as np


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class VGGBase(nn.Module):
    # Model VGG
    def __init__(self):
        super(VGGBase, self).__init__()
        model_ft = models.vgg19(pretrained=True)
        print(model_ft)
        mod = list(model_ft.features.children())
        self.lower = nn.Sequential(*mod)
        mod = list(model_ft.classifier.children())
        mod.pop()
        print(mod)
        self.upper = nn.Sequential(*mod)
        self.linear1 = nn.Linear(4096, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.linear2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)

    def forward(self, x, target=False):
        x = self.lower(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.upper(x)
        # x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))), training=False)
        # x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))), training=False)
        if target:
            return x
        else:
            return x
        return x


class VGGBase2(nn.Module):
    # Model VGG
    def __init__(self):
        super(VGGBase2, self).__init__()
        model_ft = models.vgg19(pretrained=True)
        mod = list(model_ft.classifier.children())[2:]
        mod.pop()
        print(mod)
        self.upper = nn.Sequential(*mod)
        self.linear1 = nn.Linear(4096, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.linear2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)

    def forward(self, x, target=False):
        x = self.upper(x)
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))), training=False)
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))), training=False)
        if target:
            return x
        else:
            return x


class AlexBase(nn.Module):
    def __init__(self):
        super(AlexBase, self).__init__()
        model_ft = models.alexnet(pretrained=True)
        mod = []
        print(model_ft)
        for i in range(18):
            if i < 13:
                mod.append(model_ft.features[i])
        mod_upper = list(model_ft.classifier.children())
        mod_upper.pop()
        # print(mod)
        self.upper = nn.Sequential(*mod_upper)
        self.lower = nn.Sequential(*mod)
        self.linear1 = nn.Linear(4096, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.linear2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)

    def forward(self, x, target=False, feat_return=False):
        x = self.lower(x)
        x = x.view(x.size(0), 9216)
        x = self.upper(x)
        feat = x
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))))
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))))
        if feat_return:
            return feat
        if target:
            return x
        else:
            return x


class Classifier(nn.Module):
    def __init__(self, num_classes=11):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(100, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)
        # self.fc3 = nn.Linear(100, num_classes)  # nn.Linear(100, num_classes)
        self.fc3 = nn.Linear(1000, num_classes)  # nn.Linear(100, num_classes)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if reverse:
            x = GradientReverseLayer.apply(x, self.lambd)
            # x = grad_reverse(x, self.lambd)
            feat = x
            x = self.fc3(x)
        else:
            feat = x
            x = self.fc3(x)
        if return_feat:
            return x, feat
        return x


class ResBase(nn.Module):
    def __init__(self, option='resnet18', pret=True, unit_size=100):
        super(ResBase, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)
        # default unit size 100
        self.linear1 = nn.Linear(2048, unit_size)
        self.bn1 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear2 = nn.Linear(unit_size, unit_size)
        self.bn2 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear3 = nn.Linear(unit_size, unit_size)
        self.bn3 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear4 = nn.Linear(unit_size, unit_size)
        self.bn4 = nn.BatchNorm1d(unit_size, affine=True)

    def forward(self, x,reverse=False):

        x = self.features(x)
        x = x.view(x.size(0), self.dim)
        # best with dropout
        if reverse:
            x = x.detach()

        x = F.dropout(F.relu(self.bn1(self.linear1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.linear2(x))), training=self.training)
        # x = F.dropout(F.relu(self.bn3(self.linear3(x))), training=self.training)
        # x = F.dropout(F.relu(self.bn4(self.linear4(x))), training=self.training)
        # x = F.relu(self.bn1(self.linear1(x)))
        # x = F.relu(self.bn2(self.linear2(x)))
        return x


class ResClassifier(nn.Module):
    def __init__(self, num_classes=12, unit_size=1000):
        super(ResClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(unit_size, num_classes)
        )

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_fts, out_fts):
        super(DenseBlock, self).__init__()
        self.fc = nn.Linear(in_fts, out_fts)
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_fts)

    def forward(self, x):
        x = self.fc(x)
        # x = gelu(x)
        x = self.relu(x)
        # x = F.dropout(self.relu(x))
        x = self.bn(x)
        return x


class Noise(nn.Module):
    def __init__(self, wt=0.01):
        super(Noise, self).__init__()
        self.wt = wt

    def forward(self, x, mean=0, std=1):
        x = x + self.wt*torch.randn(x.size()).cuda()
        return x


class Shallow(nn.Module):
    # Model VGG
    def __init__(self):
        super(Shallow, self).__init__()
        model_ft = models.vgg19(pretrained=True)
        mod = list(model_ft.classifier.children())[2:]
        mod.pop()
        print(mod)
        self.upper = nn.Sequential(*mod)
        # self.f1 = DenseBlock(4096, 3072)
        # self.f2 = DenseBlock(3072, 2048)
        self.linear1 = nn.Linear(4096, 3000)
        self.bn1 = nn.BatchNorm1d(3000, affine=True)
        self.linear2 = nn.Linear(3000, 2048)
        self.bn2 = nn.BatchNorm1d(2048, affine=True)
        self.noise = Noise(0.05)
        # self.noise2 = Noise(0.001)

    def forward(self, x, target=False, ep=0):
        x = self.upper(x)
        # if ep < 100:
        x = self.noise(x)
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))), training=False)
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))), training=False)
        # x = self.f1(x)
        # x = self.f2(x)
        return x


class ClassifierShadow(nn.Module):
    def __init__(self, num_classes=11):
        super(ClassifierShadow, self).__init__()
        # self.fc3 = DenseBlock(2048, 500)
        self.fc4 = DenseBlock(2048, num_classes)
        self.rev_grad = GradientReverseLayer()
        self.lambd = 1

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if reverse:
            # x = grad_reverse(x, self.lambd)
            x = self.rev_grad.apply(x, 1)
            feat = x
            # x = self.fc3(x)
            x = self.fc4(x)
        else:
            feat = x
            # x = self.fc3(x)
            x = self.fc4(x)
        if return_feat:
            return x, feat
        return x


class ShallowHome(nn.Module):
    # Model VGG
    def __init__(self, init=0):
        super(ShallowHome, self).__init__()
        self.f1 = DenseBlock(2048, 3072)
        self.f2 = DenseBlock(3072, 2048)
        if init == 1:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.fill_(0.0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, target=False):
        # x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))), training=False)
        # x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))), training=False)
        # x = F.leaky_relu(self.bn1(self.linear1(x)))
        # x = F.leaky_relu(self.bn2(self.linear2(x)))

        x = F.dropout(self.f1(x), training=False)
        x = F.dropout(self.f2(x), training=False)
        return x


class ClassifierHome(nn.Module):
    def __init__(self, num_classes=26, init=0):
        super(ClassifierHome, self).__init__()
        # self.fc3 = DenseBlock(512, 512)
        self.fc4 = DenseBlock(2048, num_classes)
        self.rev_grad = GradientReverseLayer()
        self.lambd = 1
        if init == 1:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.fill_(0.0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if reverse:
            # x = grad_reverse(x, self.lambd)
            x = self.rev_grad.apply(x, 1)
            feat = x
            # x = self.fc3(x)
            x = self.fc4(x)
        else:
            feat = x
            # x = self.fc3(x)
            x = self.fc4(x)
        if return_feat:
            return x, feat
        return x


class ShallowPie(nn.Module):
    # Model VGG
    def __init__(self, init=0):
        super(ShallowPie, self).__init__()
        self.f1 = DenseBlock(1024, 3072)
        self.f2 = DenseBlock(3072, 2048)
        if init == 1:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.fill_(0.0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, target=False):
        x = self.f1(x)
        x = self.f2(x)
        return x


class ClassifierPie(nn.Module):
    def __init__(self, num_classes=21):
        super(ClassifierPie, self).__init__()
        # self.fc3 = DenseBlock(512, 512)
        self.fc4 = DenseBlock(2048, num_classes)
        self.rev_grad = GradientReverseLayer()
        self.lambd = 1

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if reverse:
            # x = grad_reverse(x, self.lambd)
            x = self.rev_grad.apply(x, 1)
            feat = x
            # x = self.fc3(x)
            x = self.fc4(x)
        else:
            feat = x
            # x = self.fc3(x)
            x = self.fc4(x)
        if return_feat:
            return feat, x
        return x


class ShallowCLEF(nn.Module):
    def __init__(self, init=0):
        super(ShallowCLEF, self).__init__()
        self.linear1 = nn.Linear(2048, 3072)
        self.bn1 = nn.BatchNorm1d(3072, affine=True)
        self.linear2 = nn.Linear(3072, 2048)
        self.bn2 = nn.BatchNorm1d(2048, affine=True)
        # self.f1 = DenseBlock(2048, 3072)
        # self.f2 = DenseBlock(3072, 1024)
        if init == 1:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    # nn.init.kaiming_normal(m.weight, mode='fan_out')
                    m.bias.data.fill_(0.0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, target=False):
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))), training=False)
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))), training=False)
        # x = F.dropout(F.leaky_relu(self.bn3(self.linear3(x))), training=False)
        # x = self.f1(x)
        # x = self.f2(x)
        return x


class ClassifierCLEF(nn.Module):
    def __init__(self, num_classes=9):
        super(ClassifierCLEF, self).__init__()
        # self.fc3 = DenseBlock(1024, 1024)
        self.fc4 = DenseBlock(2048, num_classes)
        self.rev_grad = GradientReverseLayer()
        self.lambd = 1

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if reverse:
            # x = grad_reverse(x, self.lambd)
            x = self.rev_grad.apply(x, 1)
            feat = x
            # x = self.fc3(x)
            x = self.fc4(x)
        else:
            feat = x
            # x = self.fc3(x)
            x = self.fc4(x)
        if return_feat:
            return x, feat
        return x


class Shallow2(nn.Module):
    # Model VGG
    def __init__(self):
        super(Shallow2, self).__init__()
        self.linear1 = nn.Linear(4096, 1000)
        self.bn1 = nn.BatchNorm1d(1000, affine=True)
        self.linear2 = nn.Linear(1000, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)

    def forward(self, x, target=False):
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))), training=False)
        x = F.leaky_relu(self.bn2(self.linear2(x)))
        # x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))), training=False)
        if target:
            return x
        else:
            return x


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def calc_coeff(iter_num=0, high=1.0, low=0.0, alpha=10.0, max_iter=200.0):
    # return np.float((high - low) * (1-np.exp(-alpha*iter_num / max_iter)) + low)
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = DenseBlock(in_feature, hidden_size)
        self.ad_layer2 = DenseBlock(hidden_size, hidden_size)
        self.ad_layer3 = DenseBlock(hidden_size, 1)
        # self.dropout1 = nn.Dropout(0.5)
        # self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        # self.sft = nn.Softmax()
        # self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 200.0

    def forward(self, x, reverse=True):
        if self.training:
            self.iter_num += 1
        # coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        coeff = 1
        x = x * coeff
        if reverse:
            x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        # x = self.relu1(x)
        # x = self.dropout1(x)
        x = self.ad_layer2(x)
        # x = self.relu2(x)
        # x = self.dropout2(x)
        y = self.ad_layer3(x)
        # y = self.sft(y)[:, 0]
        y = self.sigmoid(y)
        return y
        # return x


# class AdversarialNetwork(nn.Module):
#     def __init__(self, in_feature, hidden_size):
#         super(AdversarialNetwork, self).__init__()
#         self.ad_layer1 = nn.Linear(in_feature, hidden_size)
#         self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
#         self.ad_layer3 = nn.Linear(hidden_size, 1)
#         self.relu1 = nn.LeakyReLU()
#         self.relu2 = nn.LeakyReLU()
#         self.dropout1 = nn.Dropout(0.5)
#         self.dropout2 = nn.Dropout(0.5)
#         # self.sft = nn.Softmax()
#         self.sigmoid = nn.Sigmoid()
#         # self.apply(init_weights)
#         self.iter_num = 0
#         self.alpha = 10
#         self.low = 0.0
#         self.high = 1.0
#         self.max_iter = 10000.0
#
#     def forward(self, x, reverse=True):
#         if self.training:
#             self.iter_num += 1
#         coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
#         # coeff = 1
#         x = x * coeff
#         if reverse:
#             x.register_hook(grl_hook(coeff))
#         x = self.ad_layer1(x)
#         x = self.relu1(x)
#         x = self.dropout1(x)
#         x = self.ad_layer2(x)
#         x = self.relu2(x)
#         x = self.dropout2(x)
#         y = self.ad_layer3(x)
#         y = self.sigmoid(y)
#         # y = self.sft(y)[:, 0]
#         return y
#         # return x
#
#     def output_num(self):
#         return 1
#
#     def get_parameters(self):
#         return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


if __name__ == "__main__":
    for i in range(200):
        print(calc_coeff(iter_num=i))
