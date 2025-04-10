import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import os
from tqdm.auto import tqdm
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
# from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision.utils import save_image
from functools import partial
import torch.optim as optim

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = x
        x = self.avg_pool(x)
        avg_out = self.fc(x)
        x1 = self.max_pool(x1)
        max_out = self.fc(x1)
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Feature(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256):
        super(Feature, self).__init__()
        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        self.conv1 = nn.Conv2d(kernel_size=(3, 3), in_channels=3, out_channels=8, padding=1)
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=8, out_channels=16, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(26, 1), in_channels=32, out_channels=64, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.lstm = nn.LSTM(input_size=63, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(64)
        self.gap = nn.AdaptiveAvgPool2d((None,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=64, out_features=4)
        # self.attention_query = nn.ModuleList()
        # self.attention_key = nn.ModuleList()
        # self.attention_value = nn.ModuleList()
        # for i in range(self.attention_heads):
        #     self.attention_query.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
        #     self.attention_key.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
        #     self.attention_value.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
        # 定义改动的attention
        self.attention = Attention_ssa(dim=64, num_heads=4, sr_ratio=4)  # 此时输入进来的x形状为（B,N,C）=（16,64,64），dim要与通道数对齐

    def forward(self, oa, op, on):
        x1 = self.conv1(oa)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = F.relu(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = F.relu(x1)
        x1 = self.conv4(x1)
        x1 = F.relu(x1)
        x1 = self.bn4(x1)
        residual1 = x1
        x1 = F.relu(x1)
        x1 = self.ca(x1) * x1
        x1 = self.sa(x1) * x1
        x1 += residual1
        x1 = x1.squeeze(2)
        x1, (ct, hidden) = self.lstm(x1)  #(16,64,63)-->(16,64,64)
        # x1 = x1.unsqueeze(2)
        # attn = None
        # for i in range(self.attention_heads):
        #     Q = self.attention_query[i](x1)
        #     K = self.attention_key[i](x1)
        #     V = self.attention_value[i](x1)
        #     attention = F.softmax(torch.mul(Q, K))
        #     attention = torch.mul(attention, V)
        #     #
        #     if (attn is None):
        #         attn = attention
        #     else:
        #         attn = torch.cat((attn, attention), 2)
        # x1 = attn

        # 改attention
        # x1 = x1.permute(1, 0, 2)  # x1:(B,N,C)=(16,64,32)
        x1 = self.attention(x1, H=int(np.sqrt(x1.shape[1])), W=int(np.sqrt(x1.shape[1]))) #torch.Size([16, 64, 64])
        # 此处输出的x1大小为torch.Size([16, 64, 32])

        x1 = F.relu(x1)
        x1 = self.gap(x1) #（16，64，1，1）
        x1 = x1.reshape(x1.shape[0], x1.shape[1] * x1.shape[2] * x1.shape[3]) #（16，64）

        x2 = self.conv1(op)
        x2 = self.bn1(x2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.conv3(x2)
        x2 = self.bn3(x2)
        x2 = F.relu(x2)
        x2 = self.conv4(x2)
        x2 = F.relu(x2)
        x2 = self.bn4(x2)
        residual2 = x2
        x2 = F.relu(x2)
        x2 = self.ca(x2) * x2
        x2 = self.sa(x2) * x2
        x2 += residual2
        x2 = x2.squeeze(2)
        x2, (ct, hidden) = self.lstm(x2)  #(16,64,63)-->(16,64,64)
        # x2 = x2.unsqueeze(2)
        # attn = None
        # for i in range(self.attention_heads):
        #     Q = self.attention_query[i](x2)
        #     K = self.attention_key[i](x2)
        #     V = self.attention_value[i](x2)
        #     attention = F.softmax(torch.mul(Q, K))
        #     attention = torch.mul(attention, V)
        #
        #     if (attn is None):
        #         attn = attention
        #     else:
        #         attn = torch.cat((attn, attention), 2)
        # x2 = attn

        # 改attention
        # x2 = x2.permute(1, 0, 2)  # x1:(B,N,C)=(16,64,32)
        x2 = self.attention(x2, H=int(np.sqrt(x2.shape[1])), W=int(np.sqrt(x2.shape[1]))) #torch.Size([16, 64, 64])
        # 此处输出的x1大小为torch.Size([16, 64, 64])
        x2 = F.relu(x2)
        x2 = self.gap(x2) #（16，64，1，1）
        x2 = x2.reshape(x2.shape[0], x2.shape[1] * x2.shape[2] * x2.shape[3]) #（16，64）

        x3 = self.conv1(on)
        x3 = self.bn1(x3)
        x3 = F.relu(x3)
        x3 = self.conv2(x3)
        x3 = self.bn2(x3)
        x3 = F.relu(x3)
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = self.conv4(x3)
        x3 = F.relu(x3)
        x3 = self.bn4(x3)
        residual3 = x3
        x3 = F.relu(x3)
        x3 = self.ca(x3) * x3
        x3 = self.sa(x3) * x3
        x3 += residual3
        x3 = x3.squeeze(2)
        x3, (ct, hidden) = self.lstm(x3)  #(16,64,63)-->(16,64,64)
        # x3 = x3.unsqueeze(2)
        # attn = None
        # for i in range(self.attention_heads):
        #     Q = self.attention_query[i](x3)
        #     K = self.attention_key[i](x3)
        #     V = self.attention_value[i](x3)
        #     attention = F.softmax(torch.mul(Q, K))
        #     attention = torch.mul(attention, V)
        #
        #     if (attn is None):
        #         attn = attention
        #     else:
        #         attn = torch.cat((attn, attention), 2)
        # x3 = attn
        # 改attention
        # x3 = x3.permute(1, 0, 2)  # x1:(B,N,C)=(16,64,32)
        x3 = self.attention(x3, H=int(np.sqrt(x3.shape[1])), W=int(np.sqrt(x3.shape[1]))) #torch.Size([16, 64, 64])
        # 此处输出的x1大小为torch.Size([16, 64, 32])

        x3 = F.relu(x3)
        x3 = self.gap(x3) #（16，64，1，1）
        x3 = x3.reshape(x3.shape[0], x3.shape[1] * x3.shape[2]) #（16，64）
        out1 = self.fc(x1)
        out2 = self.fc(x2)
        out3 = self.fc(x3)
        return x1, x2, x3, out1, out2, out3


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=16)
        self.Norm32 = nn.BatchNorm1d(num_features=32)
        self.Norm16 = nn.BatchNorm1d(num_features=16)
        self.fc_softmax = nn.Linear(in_features=16, out_features=4)
        self.fc_fake = nn.Linear(in_features=16, out_features=5)
        self.fc_logit = nn.Linear(in_features=16, out_features=1)
        self.softmax = nn.Softmax()

    def forward(self, *input):  # fa, fp, fn, gfa, gfp, gfn

        features_softmax = []
        features_fake = []
        features_logit = []
        for x in input:
            fa = self.fc1(x)
            fa = self.Norm32(fa)
            fa = F.relu(fa)
            fa = self.fc2(fa)
            fa = self.Norm16(fa)
            fa = F.relu(fa)
            fa = self.fc3(fa)
            fa = self.Norm16(fa)
            fa = F.relu(fa)
            fa_softmax = self.fc_softmax(fa)
            fa_fake = self.fc_fake(fa)
            fa_logit = self.fc_logit(fa)
            features_softmax.append(fa_softmax)  # torch.Size([256, 300])
            features_fake.append(fa_fake)  # torch.Size([256, 300])
            features_logit.append(fa_logit)  # torch.Size([256, 300])
        dfa_softmax = features_softmax[0]
        dfp_softmax = features_softmax[1]
        dfn_softmax = features_softmax[2]
        dga_softmax = features_softmax[3]
        dgp_softmax = features_softmax[4]
        dgn_softmax = features_softmax[5]
        dfa_fake = features_fake[0]
        dfp_fake = features_fake[1]
        dfn_fake = features_fake[2]
        dga_fake = features_fake[3]
        dgp_fake = features_fake[4]
        dgn_fake = features_fake[5]
        dfa_logit = features_logit[0]
        dfp_logit = features_logit[1]
        dfn_logit = features_logit[2]
        dga_logit = features_logit[3]
        dgp_logit = features_logit[4]
        dgn_logit = features_logit[5]

        # features_out = []
        # for x in input:
        #     fa = self.fc1(x)
        #     fa = self.Norm32(fa)
        #     fa = F.relu(fa)
        #     fa = self.fc2(fa)
        #     fa = self.Norm16(fa)
        #     fa = F.relu(fa)
        #     fa = self.fc3(fa)
        #     fa = self.Norm16(fa)
        #     fa = F.relu(fa)
        #     fa = self.fc4(fa)
        #     features_out.append(fa)  # torch.Size([256, 300])
        # # dfa = features_out[0]
        # dfa = features_out[0]
        # dfp = features_out[1]
        # dfn = features_out[2]

        return dfa_softmax, dfp_softmax, dfn_softmax, dga_softmax, dgp_softmax, dgn_softmax, dfa_fake, dfp_fake, dfn_fake, dga_fake, dgp_fake, dgn_fake, dfa_logit, dfp_logit, dfn_logit, dga_logit, dgp_logit, dgn_logit


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=64)
        self.Norm32 = nn.BatchNorm1d(num_features=32)
        self.Norm16 = nn.BatchNorm1d(num_features=16)
        self.Norm64 = nn.BatchNorm1d(num_features=64)

    def forward(self, *input):
        xa = input[0]
        xp = input[1]
        xn = input[2]
        features = []
        for x in input:
            fa = self.fc1(x)
            fa = self.Norm32(fa)
            fa = F.relu(fa)
            fa = self.fc2(fa)
            fa = self.Norm16(fa)
            fa = F.relu(fa)
            fa = self.fc3(fa)
            fa = self.Norm32(fa)
            fa = F.relu(fa)
            fa = self.fc4(fa)
            fa = self.Norm64(fa)
            fa = F.relu(fa)
            features.append(fa)

        ga = xa + features[0]
        gp = xp + features[1]
        gn = xn + features[2]

        return ga, gp, gn


class Net(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256):
        super(Net, self).__init__()
        self.F = Feature(attention_heads, attention_hidden)
        self.D = Discriminator()
        self.G = Generator()
        self.fc = nn.Linear(in_features=5, out_features=4)

    def forward(self, *input):
        oa = input[0]
        op = input[1]
        on = input[2]
        fa, fp, fn, out1, out2, out3 = self.F(oa, op, on)
        gfa, gfp, gfn = self.G(fa, fp, fn)
        dfa, dfp, dfn, dgfa, dgfp, dgfn, dgfa_fake, dgfp_fake, dgfn_fake, = self.D(fa, fp, fn, gfa, gfp, gfn)

        return gfa, gfp, gfn, dfa, dfp, dfn, dgfa, dgfp, dgfn, dgfa_fake, dgfp_fake, dgfn_fake, out1, out2, out3


class Encoder(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256):
        super(Encoder, self).__init__()
        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        self.conv1 = nn.Conv2d(kernel_size=(3, 3), in_channels=3, out_channels=8, padding=1)
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=8, out_channels=16, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(26, 1), in_channels=32, out_channels=64, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.lstm = nn.LSTM(input_size=63, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()
        self.fc = nn.Linear(in_features=64, out_features=4)
        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))

    def forward(self, oa):
        x1 = self.conv1(oa)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = F.relu(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = F.relu(x1)
        x1 = self.conv4(x1)
        x1 = F.relu(x1)
        x1 = self.bn4(x1)
        residual1 = x1
        x1 = F.relu(x1)
        x1 = self.ca(x1) * x1
        x1 = self.sa(x1) * x1
        x1 += residual1
        x1 = x1.squeeze(2)
        x1, (ct, hidden) = self.lstm(x1)
        x1 = x1.unsqueeze(2)  #(16,64,64)-->(16,64,1,64)
        attn = None
        for i in range(self.attention_heads):
            Q = self.attention_query[i](x1)
            K = self.attention_key[i](x1)
            V = self.attention_value[i](x1)
            attention = F.softmax(torch.mul(Q, K))
            attention = torch.mul(attention, V)
            #
            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x1 = attn          #(16,64,4,64)
        x1 = F.relu(x1)
        x1 = self.gap(x1)  #(16,64,4,64)-->(16,64,1,1)
        embeddings = x1.reshape(x1.shape[0], x1.shape[1] * x1.shape[2] * x1.shape[3])

        return embeddings


class Encoder1(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256):
        super(Encoder1, self).__init__()
        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        self.conv1 = nn.Conv2d(kernel_size=(3, 3), in_channels=3, out_channels=8, padding=1)
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=8, out_channels=16, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(26, 1), in_channels=32, out_channels=64, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.lstm = nn.LSTM(input_size=63, hidden_size=32, num_layers=1, bidirectional=True, batch_first=True)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=64, out_features=4)
        # self.attention_query = nn.ModuleList()
        # self.attention_key = nn.ModuleList()
        # self.attention_value = nn.ModuleList()
        # for i in range(self.attention_heads):
        #     self.attention_query.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
        #     self.attention_key.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
        #     self.attention_value.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
        # 定义改动的attention
        self.attention = Attention_ssa(dim=64, num_heads=4, sr_ratio=4)  # 此时输入进来的x形状为（B,N,C）=（16,64,64），dim要与通道数对齐

    def forward(self, oa):
        x1 = self.conv1(oa)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = F.relu(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = F.relu(x1)
        x1 = self.conv4(x1)
        x1 = F.relu(x1)
        x1 = self.bn4(x1)
        residual1 = x1
        x1 = F.relu(x1)
        x1 = self.ca(x1) * x1
        x1 = self.sa(x1) * x1
        x1 += residual1
        x1 = x1.squeeze(2)
        x1, (ct, hidden) = self.lstm(x1)  #(16,64,63)-->(16,64,64)
        # x1 = x1.unsqueeze(2)
        # attn = None
        # for i in range(self.attention_heads):
        #     Q = self.attention_query[i](x1)
        #     K = self.attention_key[i](x1)
        #     V = self.attention_value[i](x1)
        #     attention = F.softmax(torch.mul(Q, K))
        #     attention = torch.mul(attention, V)
        #     #
        #     if (attn is None):
        #         attn = attention
        #     else:
        #         attn = torch.cat((attn, attention), 2)
        # x1 = attn

        # 改attention
        # x1 = x1.permute(1, 0, 2)  # (16,64,64)--> x1:(B,N,C)=(64,16,64)
        x1 = self.attention(x1, H=int(np.sqrt(x1.shape[1])), W=int(np.sqrt(x1.shape[1])))  #(16,64,64)
        # 此处输出的x1大小为torch.Size([16, 64, 32])

        x1 = F.relu(x1)
        x1 = x1.unsqueeze(2)
        x1 = self.gap(x1)  #（16,64,1,1）
        embeddings = x1.reshape(x1.shape[0], x1.shape[1] * x1.shape[2])  #（16,64）

        return embeddings






class Generator1(nn.Module):
    def __init__(self):
        super(Generator1, self).__init__()
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=64)
        self.Norm32 = nn.BatchNorm1d(num_features=32)
        self.Norm16 = nn.BatchNorm1d(num_features=16)
        self.Norm64 = nn.BatchNorm1d(num_features=64)

    def forward(self, x):
        fa = self.fc1(x)  #（16，64）
        fa = self.Norm32(fa)
        fa = F.relu(fa)
        fa = self.fc2(fa)
        fa = self.Norm16(fa)
        fa = F.relu(fa)
        fa = self.fc3(fa)
        fa = self.Norm32(fa)
        fa = F.relu(fa)
        fa = self.fc4(fa)
        fa = self.Norm64(fa)
        gfa = F.relu(fa)
        return gfa


class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=16)
        self.fc5 = nn.Linear(in_features=16, out_features=4)
        self.fc4 = nn.Linear(in_features=4, out_features=1)
        self.Norm32 = nn.BatchNorm1d(num_features=32)
        self.Norm16 = nn.BatchNorm1d(num_features=16)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fa = self.fc1(x)     #(16,64)
        fa = self.Norm32(fa)
        fa = F.relu(fa)
        fa = self.fc2(fa)
        fa = self.Norm16(fa)
        fa = F.relu(fa)
        fa = self.fc3(fa)
        fa = self.Norm16(fa)
        fa = F.relu(fa)
        d_embeddings4 = self.fc5(fa)
        d_embeddings1 = self.fc4(d_embeddings4)
        # d_embeddings = self.sigmoid(fa)
        # return d_embeddings1
        return d_embeddings1, d_embeddings4   #(16,1),(16,4)


class Classifiar(nn.Module):
    def __init__(self):
        super(Classifiar, self).__init__()
        self.fc = nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        output = self.fc(x)
        return output

#SSA注意力模块

def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

class Attention_ssa(nn.Module):
    # 这段代码定义了一个名为Attention的类，它是神经网络中的一个模块，实现了自注意力机制。
    #
    # 代码的主要功能如下：
    #
    # __init__方法初始化了注意力模块的参数和层。它接收参数，例如dim（输入的维度）、num_heads（注意力头的数量）、qkv_bias（是否在线性层中使用偏置项）、qk_scale（查询 - 键之间点积的缩放因子）、attn_drop（注意力权重的dropout率）、proj_drop（输出的dropout率）和sr_ratio（子采样比率）。
    #
    # _init_weights方法用于初始化模块中线性层和卷积层的权重。
    #
    # forward方法执行注意力模块的前向传播。它接收输入x（形状为[B, N, C]），其中B是批次大小，N是序列中的元素数量，C是输入的维度。
    #
    # 在forward方法内部：
    #
    # 输入x经过线性层（self.q）变换，得到查询向量（q）。
    # 如果sr_ratio大于1，则将输入x重塑并通过一系列卷积层和线性层处理，以提取不同的特征图。这样做是为了捕捉不同尺度上的信息。
    # 查询向量（q）与键向量（k）进行点积，并乘以头维度的平方根，得到注意力权重。
    # 注意力权重被应用于值向量（v），得到注意力的输出。
    # 最后，输出经过线性层和dropout层处理后返回。
    # 综上所述，该代码实现了自注意力机制，用于学习输入序列中的元素之间的关系，并生成相应的注意力表示。
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio==8:
                # 通过两次不同尺寸大小的卷积核来提取不同的feature map 8*8， 4*4，
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==4:
                # 通过两次不同尺寸大小的卷积核来提取不同的feature map 4*4,2*2
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==2:
                # 通过两次不同尺寸大小的卷积核来提取不同的feature map 2*2，1*1
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
                self.norm2 = nn.LayerNorm(dim)
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv1 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
            self.local_conv2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # B：代表批次（Batch）的维度。
        # N：代表样本数量（Number）的维度。
        # C：代表通道数（Channel）的维度
        B, N, C = x.shape
       # 这个reshape操作将查询结果重塑为一个四维张量，其中第一维表示批次，第二维表示注意力头，第三维表示样本数量，第四维表示通道数。
       #  其中的参数(0, 2, 1, 3)
       #  表示将张量的维度重新排列为(0, 2, 1, 3)
       #  的顺序。这个操作是为了将样本数量和注意力头的维度进行交换。
       #  结果是一个重排后的四维张量，其中第一维表示批次，第二维表示样本数量，第三维表示注意力头，第四维表示通道数。
       #  整体来说，这行代码的目的是将输入张量
       #  通过查询操作，并根据注意力头的数量和通道数进行重塑和维度置换，以便后续进行自注意力计算。
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0], kv1[1] #B head N C
                k2, v2 = kv2[0], kv2[1]
                attn1 = (q[:, :self.num_heads//2] @ k1.transpose(-2, -1)) * self.scale
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).
                                        transpose(1, 2).view(B,C//2, H//self.sr_ratio, W//self.sr_ratio)).\
                    view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)
                attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).
                                        transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio)).\
                    view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)

                x = torch.cat([x1,x2], dim=-1)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N, C).
                                        transpose(1, 2).view(B,C, H, W)).view(B, C, N).transpose(1, 2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x






#扩散模型模块
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def exists(x):
    return x is not None


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class GaussianDiffusionBlock(nn.Module):
    def __init__(self, model, image_size, timesteps=1000, beta_schedule='linear', auto_normalize=True):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels

        self.image_size = image_size

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # calculate beta and other precalculated parameters
        betas = beta_schedule_fn(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = timesteps  # default num sampling timesteps to number of timesteps at training

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()

        register_buffer('loss_weight', maybe_clipped_snr / snr)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, clip_x_start=False, rederive_pred_noise=False):
        model_output = self.model(x, t)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)

        if clip_x_start and rederive_pred_noise:
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def p_mean_variance(self, x, t, clip_denoised=True):
        noise, x_start = self.model_predictions(x, t)

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, clip_denoised=True)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps=False, save_dir=None):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        #         torch.randn_like(torch.empty(shape), device=device)
        imgs = [img]

        x_start = None

        ###########################################
        ## TODO: plot the sampling process ##
        ###########################################

        # 保存每个时间步的采样结果
        all_samples = [img]

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(img, t)
            imgs.append(img)
            all_samples.append(img)

            # 保存每个时间步的采样结果
            if save_dir is not None:
                # 将图像保存为文件
                file_path = os.path.join(save_dir, f'step_{t}.png')
                save_image(self.unnormalize(img), file_path)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret, all_samples

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps=return_all_timesteps)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, img, t: int):
        """
        Forward pass for the diffusion block.

        Args:
            img (torch.Tensor): Input image tensor. Shape should be (batch_size, channels, image_size, image_size).
            t (int): The time step for diffusion.

        Returns:
            torch.Tensor: Sampled image tensor. Shape is same as input image tensor.
        """
        b, c, h, w = img.shape
        assert h == self.image_size and w == self.image_size, f'height and width of image must be {self.image_size}'
        img = self.normalize(img)
        return self.p_sample(img, t)


class CustomRNN(nn.Module):   #改进的BLSTM模块
    def __init__(self,input_size,hidden_size,output_size):
        super(CustomRNN, self).__init__()

        self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=hidden_size, hidden_size=output_size, num_layers=1, batch_first=True)
        self.max_pooling = nn.MaxPool1d(kernel_size=1)

    def forward(self, x):
        # RNN1          #输入x的形状为（B,N,C）=（16，64，63）
        output_rnn1, _ = self.rnn1(x)  #（B,N,C）=（16，64，128）

        # Reshape the output of RNN1 to create segments
        segment_length = 4
        num_segments = output_rnn1.size(1) // segment_length
        output_rnn1_reshaped = output_rnn1.view(-1, num_segments, segment_length, 128)  #（16，16，4，128）

        # Take the last step of each segment as the representative feature vector
        rnn1_last_step_features = output_rnn1_reshaped[:, :, -1, :]                    #（16，16，128）

        # Create shifted views
        shifted_rnn1_3 = torch.cat([rnn1_last_step_features[:, i::13] for i in range(13)], dim=1)  #（16，16，128）
        shifted_rnn1_6 = torch.cat([rnn1_last_step_features[:, i::16] for i in range(16)], dim=1)  #（16，16，128）
        shifted_rnn1_9 = torch.cat([rnn1_last_step_features[:, i::19] for i in range(19)], dim=1)  #（16，16，128）

        # RNN2 for original view
        output_rnn2_1, _ = self.rnn2(rnn1_last_step_features) #（16，16，256）

        # RNN2 for shifted view 3
        output_rnn2_3, _ = self.rnn2(shifted_rnn1_3)          #（16，16，256）

        # RNN2 for shifted view 6
        output_rnn2_6, _ = self.rnn2(shifted_rnn1_6)          #（16，16，256）

        # RNN2 for shifted view 9
        output_rnn2_9, _ = self.rnn2(shifted_rnn1_9)          #（16，16，256）

        # Max pooling over segments for all views
        # pooled_rnn2_1 = self.max_pooling(output_rnn2_1.transpose(1, 2)).transpose(1, 2) #(16,16,256)
        # pooled_rnn2_3 = self.max_pooling(output_rnn2_3.transpose(1, 2)).transpose(1, 2)
        # pooled_rnn2_6 = self.max_pooling(output_rnn2_6.transpose(1, 2)).transpose(1, 2)
        # pooled_rnn2_9 = self.max_pooling(output_rnn2_9.transpose(1, 2)).transpose(1, 2)

        output = torch.cat([output_rnn2_1, output_rnn2_3, output_rnn2_6, output_rnn2_9], dim=1)



        return output  #(16,64,256)


class Encoder_Dm(nn.Module):    #更适合于扩散模型的encorder，需要将输出的维度进行一定的调整，扩充
    def __init__(self, attention_heads=8, attention_hidden=256):
        super(Encoder_Dm, self).__init__()
        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        self.conv1 = nn.Conv2d(kernel_size=(3, 3), in_channels=3, out_channels=8, padding=1)
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=8, out_channels=16, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(26, 1), in_channels=32, out_channels=64, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.lstm = nn.LSTM(input_size=63, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
        self.blstm = CustomRNN(input_size=63, hidden_size=128,output_size=256)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features=64, out_features=4)
        # self.attention_query = nn.ModuleList()
        # self.attention_key = nn.ModuleList()
        # self.attention_value = nn.ModuleList()
        # for i in range(self.attention_heads):
        #     self.attention_query.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
        #     self.attention_key.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
        #     self.attention_value.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
        # 定义改动的attention
        self.attention = Attention_ssa(dim=64, num_heads=4, sr_ratio=4)  # 此时输入进来的x形状为（B,N,C）=（16,256,64），dim要与通道数对齐


    def forward(self, oa):
        x1 = self.conv1(oa)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = F.relu(x1)
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = F.relu(x1)
        x1 = self.conv4(x1)
        x1 = F.relu(x1)
        x1 = self.bn4(x1)
        residual1 = x1
        x1 = F.relu(x1)
        x1 = self.ca(x1) * x1
        x1 = self.sa(x1) * x1
        x1 += residual1
        x1 = x1.squeeze(2)
        # x1, (ct, hidden) = self.lstm(x1)  #(16,64,63)-->(16,64,256)
        x1= self.blstm(x1)
        # x1 = x1.unsqueeze(2)
        # attn = None
        # for i in range(self.attention_heads):
        #     Q = self.attention_query[i](x1)
        #     K = self.attention_key[i](x1)
        #     V = self.attention_value[i](x1)
        #     attention = F.softmax(torch.mul(Q, K))
        #     attention = torch.mul(attention, V)
        #     #
        #     if (attn is None):
        #         attn = attention
        #     else:
        #         attn = torch.cat((attn, attention), 2)
        # x1 = attn

        # 改attention
        x1 = x1.permute(0, 2, 1)  # (16,64,256)--> x1:(B,N,C)=(16,256,64)
        x1 = self.attention(x1, H=int(np.sqrt(x1.shape[1])), W=int(np.sqrt(x1.shape[1])))  #x1:(B,N,C)=(16,256,64)
        # 此处输出的x1大小为torch.Size([16, 256, 64])
        # x1 = x1.permute(0, 2, 1)  #将维度修改回来改为(16,64,256)

        x1 = F.relu(x1)
        x1 = x1.unsqueeze(2)
        x1 = self.gap(x1)  #（16,256,1,1）
        embeddings = x1.reshape(x1.shape[0], x1.shape[1] * x1.shape[2])  #（16,256）

        return embeddings



class Generator_Dm(nn.Module):
    def __init__(self, model, image_size, timesteps=100, beta_schedule='linear', auto_normalize=True):
        super().__init__()
        self.diffusion_block = GaussianDiffusionBlock(model, image_size, timesteps, beta_schedule, auto_normalize)

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False):
        image_size, channels = self.diffusion_block.image_size, self.diffusion_block.channels
        sample_fn = self.diffusion_block.p_sample_loop
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps=return_all_timesteps)

    def forward(self, img, t: int):
        img = img.view(16,4,8,8)
        b, c, h, w = img.shape
        assert h == self.diffusion_block.image_size and w == self.diffusion_block.image_size, f'height and width of image must be {self.diffusion_block.image_size}'
        img = self.diffusion_block.normalize(img)

        # fake_samples, _ = generator.sample(batch_size=len(real_samples), return_all_timesteps=False)
        return self.diffusion_block.p_sample(img, t)


class Discriminator_Dm(nn.Module):
    def __init__(self):
        super(Discriminator_Dm, self).__init__()
        self.fc0 = nn.Linear(in_features=256, out_features=128)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
        self.fc4 = nn.Linear(in_features=16, out_features=16)
        self.fc5 = nn.Linear(in_features=16, out_features=4)
        self.fc6 = nn.Linear(in_features=4, out_features=1)
        self.Norm128 = nn.BatchNorm1d(num_features=128)
        self.Norm64 = nn.BatchNorm1d(num_features=64)
        self.Norm32 = nn.BatchNorm1d(num_features=32)
        self.Norm16 = nn.BatchNorm1d(num_features=16)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if len(x.size()) == 4:
            b, c, h, w = x.size()
            x = x.view(b, c * h * w)  # 修改维度为(16,256)
        fa = self.fc0(x)
        fa = self.Norm128(fa)
        fa = F.relu(fa)
        fa = self.fc1(fa)
        fa = self.Norm64(fa)
        fa = F.relu(fa)
        fa = self.fc2(fa)
        fa = self.Norm32(fa)
        fa = F.relu(fa)
        fa = self.fc3(fa)
        fa = self.Norm16(fa)
        fa = F.relu(fa)
        fa = self.fc4(fa)
        fa = self.Norm16(fa)
        fa = F.relu(fa)
        d_embeddings4 = self.fc5(fa)
        d_embeddings1 = self.fc6(d_embeddings4)
        # d_embeddings = self.sigmoid(fa)
        # return d_embeddings1
        return d_embeddings1, d_embeddings4  #(16,1),(16,4)

class Classifiar_Dm(nn.Module):
    def __init__(self):
        super(Classifiar_Dm, self).__init__()
        self.fc0 = nn.Linear(in_features=256, out_features=64)
        self.fc1 = nn.Linear(in_features=64, out_features=4)
        self.Norm64 = nn.BatchNorm1d(num_features=64)

    def forward(self, x):
        x = self.fc0(x)
        x = self.Norm64(x)
        x = F.relu(x)
        output = self.fc1(x)
        return output

# # 两层全连接网络作为分类器
# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.fc0 = nn.Linear(in_features=256, out_features=64)
#         self.fc1 = nn.Linear(in_features=64, out_features=4)
#         self.Norm64 = nn.BatchNorm1d(num_features=64)
#
#     def forward(self, x):
#         x = self.fc0(x)
#         x = self.Norm64(x)
#         x = F.relu(x)
#         output = self.fc1(x)
#         return output
#
#     # 重写train方法
#     def train(self, features, labels,num):
#         num_epochs = 1
#         learning_rate = 0.001
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(self.Classifier.parameters(), lr=learning_rate,weight_decay=1e-6)
#         # 训练循环
#         for epoch in range(num_epochs):
#             # 前向传播
#             # x = self.fc1(features)
#             # x = F.relu(x)
#             # outputs = self.fc2(x)
#             outputs = self(features)
#             loss = criterion(outputs, labels)
#             # 打印分类loss
#             print(f"分类器损失：Epoch {num + 1}, Class Loss: {loss.item():.4f}")
#
#             # 反向传播
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#     # 重写eval方法
#     def eval(self):
#         # 设置eval模式
#         self.training = False
#         for module in self.children():
#             module.training = False
#
#         return self

class Classifiar_Dm(nn.Module):
    def __init__(self):
        super(Classifiar_Dm, self).__init__()
        self.fc0 = nn.Linear(in_features=256, out_features=128)
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=4)
        self.Norm128 = nn.BatchNorm1d(num_features=128)
        self.Norm64 = nn.BatchNorm1d(num_features=64)
        self.Norm32 = nn.BatchNorm1d(num_features=32)

    def forward(self, x):
        x = self.fc0(x)
        x = self.Norm128(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = self.Norm64(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.Norm32(x)
        x = F.relu(x)
        output = self.fc3(x)
        return output


#定义 MV-BiSSM
class CustomSSM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, ssm_block):
        super().__init__()
        self.ssm1 = ssm_block(input_size, hidden_size)
        self.ssm2 = ssm_block(hidden_size, output_size)

    def forward(self, x):
        out1 = self.ssm1(x)

        segment_len = 4
        num_segments = out1.size(1) // segment_len
        reshaped = out1.view(-1, num_segments, segment_len, out1.size(2))
        last_step = reshaped[:, :, -1, :]

        v3 = torch.cat([last_step[:, i::13] for i in range(13)], dim=1)
        v6 = torch.cat([last_step[:, i::16] for i in range(16)], dim=1)
        v9 = torch.cat([last_step[:, i::19] for i in range(19)], dim=1)

        out2_1 = self.ssm2(last_step)
        out2_3 = self.ssm2(v3)
        out2_6 = self.ssm2(v6)
        out2_9 = self.ssm2(v9)

        return torch.cat([out2_1, out2_3, out2_6, out2_9], dim=1)


# #SER3方法
# from s4torch import S4  # 需要安装 s4torch 库 https://github.com/TariqAHassan/S4Torch
#
# # S4 Block
# class S4Block(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(S4Block, self).__init__()
#         self.input_proj = nn.Linear(input_size, hidden_size)
#         self.s4 = S4(hidden_size)
#
#     def forward(self, x):
#         x_proj = self.input_proj(x)
#         return self.s4(x_proj)
#
# #封装成双向结构（双向 CustomSSM）
# class BidirectionalCustomSSM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, ssm_block):
#         super().__init__()
#         self.custom_fwd = CustomSSM(input_size, hidden_size, output_size, ssm_block)
#         self.custom_bwd = CustomSSM(input_size, hidden_size, output_size, ssm_block)
#
#     def forward(self, x):
#         x_rev = torch.flip(x, dims=[1])
#         y_fwd = self.custom_fwd(x)
#         y_bwd = self.custom_bwd(x_rev)
#         y_bwd = torch.flip(y_bwd, dims=[1])
#         return torch.cat([y_fwd, y_bwd], dim=-1)
#
# # 集成到Mamba中
# class MambaBlock(nn.Module):
#     def __init__(self, input_size, hidden_size, ssm_layer):
#         super().__init__()
#         self.layer_norm = nn.LayerNorm(input_size)
#         self.ssm = ssm_layer  # ← 就是我们要换为我们方法的地方
#         self.linear = nn.Linear(hidden_size * 2, input_size)  # 双向合并回去
#
#     def forward(self, x):
#         x_norm = self.layer_norm(x)
#         x_ssm = self.ssm(x_norm)
#         return self.linear(x_ssm) + x  # 残差连接
#
# class Discriminator_Dm_SER3(nn.Module):
#     def __init__(self):
#         super(Discriminator_Dm_SER3, self).__init__()
#         self.fc0 = nn.Linear(in_features=256, out_features=128)
#         self.fc1 = nn.Linear(in_features=128, out_features=64)
#         self.fc2 = nn.Linear(in_features=64, out_features=32)
#         self.fc3 = nn.Linear(in_features=32, out_features=16)
#         self.fc4 = nn.Linear(in_features=16, out_features=16)
#         self.fc5 = nn.Linear(in_features=16, out_features=4)
#         self.fc6 = nn.Linear(in_features=4, out_features=1)
#         self.Norm128 = nn.BatchNorm1d(num_features=128)
#         self.Norm64 = nn.BatchNorm1d(num_features=64)
#         self.Norm32 = nn.BatchNorm1d(num_features=32)
#         self.Norm16 = nn.BatchNorm1d(num_features=16)
#         # self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         if len(x.size()) == 4:
#             b, c, h, w = x.size()
#             x = x.view(b, c * h * w)  # 修改维度为(16,256)
#         fa = self.fc0(x)
#         fa = self.Norm128(fa)
#         fa = F.relu(fa)
#         fa = self.fc1(fa)
#         fa = self.Norm64(fa)
#         fa = F.relu(fa)
#         fa = self.fc2(fa)
#         fa = self.Norm32(fa)
#         fa = F.relu(fa)
#         fa = self.fc3(fa)
#         fa = self.Norm16(fa)
#         fa = F.relu(fa)
#         fa = self.fc4(fa)
#         fa = self.Norm16(fa)
#         fa = F.relu(fa)
#         d_embeddings4 = self.fc5(fa)
#         d_embeddings1 = self.fc6(d_embeddings4)
#         # d_embeddings = self.sigmoid(fa)
#         # return d_embeddings1
#         return d_embeddings1, d_embeddings4  #(16,1),(16,4)
#
# class Encoder_Dm_SER3(nn.Module):    #更适合于扩散模型的encorder，需要将输出的维度进行一定的调整，扩充
#     def __init__(self, attention_heads=8, attention_hidden=256):
#         super(Encoder_Dm_SER3, self).__init__()
#         self.attention_heads = attention_heads
#         self.attention_hidden = attention_hidden
#         self.conv1 = nn.Conv2d(kernel_size=(3, 3), in_channels=3, out_channels=8, padding=1)
#         self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=8, out_channels=16, padding=1)
#         self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
#         self.conv4 = nn.Conv2d(kernel_size=(26, 1), in_channels=32, out_channels=64, padding=0)
#         self.bn1 = nn.BatchNorm2d(8)
#         self.bn2 = nn.BatchNorm2d(16)
#         self.bn3 = nn.BatchNorm2d(32)
#         self.bn4 = nn.BatchNorm2d(64)
#         # self.lstm = nn.LSTM(input_size=63, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)
#         # self.blstm = CustomRNN(input_size=63, hidden_size=128,output_size=256)
#         self.ssm_block = lambda in_dim, out_dim: S4Block(input_size=63, hidden_size=128)   #可以替换为其他的SSM逻辑为多视图结构的基础SSM，例如：Mamba、S3等
#         self.bi_custom_ssm = BidirectionalCustomSSM(input_size=63,hidden_size=128,output_size=256,ssm_block=self.ssm_block)
#         self.mamba_layer = MambaBlock(input_size=63,hidden_size=256,ssm_layer=self.bi_custom_ssm)
#         self.sa = SpatialAttention()
#         self.ca = ChannelAttention(64)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.dropout = nn.Dropout(0.5)
#         self.fc = nn.Linear(in_features=64, out_features=4)
#         # self.attention_query = nn.ModuleList()
#         # self.attention_key = nn.ModuleList()
#         # self.attention_value = nn.ModuleList()
#         # for i in range(self.attention_heads):
#         #     self.attention_query.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
#         #     self.attention_key.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
#         #     self.attention_value.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1))
#         # 定义改动的attention
#         self.attention = Attention_ssa(dim=64, num_heads=4, sr_ratio=4)  # 此时输入进来的x形状为（B,N,C）=（16,256,64），dim要与通道数对齐
#
#
#     def forward(self, oa):
#         x1 = self.conv1(oa)
#         x1 = self.bn1(x1)
#         x1 = F.relu(x1)
#         x1 = self.conv2(x1)
#         x1 = self.bn2(x1)
#         x1 = F.relu(x1)
#         x1 = self.conv3(x1)
#         x1 = self.bn3(x1)
#         x1 = F.relu(x1)
#         x1 = self.conv4(x1)
#         x1 = F.relu(x1)
#         x1 = self.bn4(x1)
#         residual1 = x1
#         x1 = F.relu(x1)
#         x1 = self.ca(x1) * x1
#         x1 = self.sa(x1) * x1
#         x1 += residual1
#         x1 = x1.squeeze(2)
#         # x1, (ct, hidden) = self.lstm(x1)  #(16,64,63)-->(16,64,256)
#         x1= self.mamba_layer(x1)
#         # x1 = x1.unsqueeze(2)
#         # attn = None
#         # for i in range(self.attention_heads):
#         #     Q = self.attention_query[i](x1)
#         #     K = self.attention_key[i](x1)
#         #     V = self.attention_value[i](x1)
#         #     attention = F.softmax(torch.mul(Q, K))
#         #     attention = torch.mul(attention, V)
#         #     #
#         #     if (attn is None):
#         #         attn = attention
#         #     else:
#         #         attn = torch.cat((attn, attention), 2)
#         # x1 = attn
#
#         # 改attention
#         x1 = x1.permute(0, 2, 1)  # (16,64,256)--> x1:(B,N,C)=(16,256,64)
#         x1 = self.attention(x1, H=int(np.sqrt(x1.shape[1])), W=int(np.sqrt(x1.shape[1])))  #x1:(B,N,C)=(16,256,64)
#         # 此处输出的x1大小为torch.Size([16, 256, 64])
#         # x1 = x1.permute(0, 2, 1)  #将维度修改回来改为(16,64,256)
#
#         x1 = F.relu(x1)
#         x1 = x1.unsqueeze(2)
#         x1 = self.gap(x1)  #（16,256,1,1）
#         embeddings = x1.reshape(x1.shape[0], x1.shape[1] * x1.shape[2])  #（16,256）
#
#         return embeddings
#
#
# #根据输入数据的量选择合适的分类器，少量数据选层数少的，大量数据选择层数多的
# class Classifiar_Dm_SER3(nn.Module):
#     def __init__(self):
#         super(Classifiar_Dm_SER3, self).__init__()
#         self.fc0 = nn.Linear(in_features=256, out_features=64)
#         self.fc1 = nn.Linear(in_features=64, out_features=4)
#         self.Norm64 = nn.BatchNorm1d(num_features=64)
#
#     def forward(self, x):
#         x = self.fc0(x)
#         x = self.Norm64(x)
#         x = F.relu(x)
#         output = self.fc1(x)
#         return output
#
# # class Classifiar_Dm_SER3(nn.Module):
# #     def __init__(self):
# #         super(Classifiar_Dm_SER3, self).__init__()
# #         self.fc0 = nn.Linear(in_features=256, out_features=128)
# #         self.fc1 = nn.Linear(in_features=128, out_features=64)
# #         self.fc2 = nn.Linear(in_features=64, out_features=32)
# #         self.fc3 = nn.Linear(in_features=32, out_features=4)
# #         self.Norm128 = nn.BatchNorm1d(num_features=128)
# #         self.Norm64 = nn.BatchNorm1d(num_features=64)
# #         self.Norm32 = nn.BatchNorm1d(num_features=32)
# #
# #     def forward(self, x):
# #         x = self.fc0(x)
# #         x = self.Norm128(x)
# #         x = F.relu(x)
# #         x = self.fc1(x)
# #         x = self.Norm64(x)
# #         x = F.relu(x)
# #         x = self.fc2(x)
# #         x = self.Norm32(x)
# #         x = F.relu(x)
# #         output = self.fc3(x)
# #         return output







