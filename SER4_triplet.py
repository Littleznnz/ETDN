import glob
import os
import pickle
import random
import time
import math
import logging
import datetime
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
# import librosa
from tqdm import tqdm
import os
# import features
import net
import data_loader
import pandas as pd
from net import Discriminator_Dm_SER3,Encoder_Dm_SER3,Classifiar_Dm_SER3 #从网络中找到扩散模型定义的块
from sklearn.model_selection import train_test_split
import math
import warnings
# from s4torch import S4  # 需要安装 s4torch 库 https://github.com/TariqAHassan/S4Torch 用来加载SSM4模型

plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置指定使用第 n 块GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#自定义去噪神经网络
class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=256):
        super(MLPDiffusion, self).__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(256, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 256),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

    def forward(self, x, t):
        #         x = x_0
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t).cuda()
            x = self.linears[2 * idx](x) #选取的是线性层
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)

        return x

#编写训练误差函数
def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """对任意时刻t进行采样计算loss"""
    batch_size = x_0.shape[0]

    # 对一个batchsize样本生成随机的时刻t，t变得随机分散一些，一个batch size里面覆盖更多的t
    t = torch.randint(0, n_steps, size=(batch_size // 2,))
    t = torch.cat([t, n_steps - 1 - t], dim=0)# t的形状（bz）
    t = t.unsqueeze(-1)# t的形状（bz,1）
    t = t.cuda()

    # x0的系数，根号下(alpha_bar_t)
    a = alphas_bar_sqrt[t].cuda()

    # eps的系数,根号下(1-alpha_bar_t)
    aml = one_minus_alphas_bar_sqrt[t].cuda()

    # 生成随机噪音eps
    e = torch.randn_like(x_0).cuda()

    # 构造模型的输入
    x = x_0 * a + e * aml
    x = x.cuda()

    # 送入模型，得到t时刻的随机噪声预测值
    output = model(x, t.squeeze(-1))

    # 与真实噪声一起计算误差，求平均值
    return (e - output).square().mean()

#编写逆扩散采样函数

#从x_t恢复x_0  x_t是随机噪声
def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
    cur_x = torch.randn(shape).cuda()
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq

#根据预测出来的噪声的均值以及方差计算当前时刻的数据分布
def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    """从x[T]采样t时刻的重构值"""
    t = torch.tensor([t]).cuda()

    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]

    eps_theta = model(x, t)
    #得到均值
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    #得到sample的分布
    sample = mean + sigma_t * z

    return (sample)

#开始训练模型，打印loss以及中间的重构效果
seed = 1234


class EMA():
    """构建一个参数平滑器"""

    def __init__(self, mu=0.01):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average

#确定超参数的值
num_steps = 100  #可由beta值估算

#制定每一步的beta，beta按照时间从小到大变化
betas = torch.linspace(-6,6,num_steps)
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5
betas = betas.cuda()
#计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
alphas = 1-betas
alphas = alphas.cuda()
# alpha连乘
alphas_prod = torch.cumprod(alphas,0)
alphas_prod = alphas_prod.cuda()
#从第一项开始，第0项另乘1？？？
alphas_prod_p = torch.cat([torch.tensor([1]).float().cuda(),alphas_prod[:-1]],0)
alphas_prod_p = alphas_prod_p.cuda()
# alphas_prod开根号
alphas_bar_sqrt = torch.sqrt(alphas_prod)
alphas_bar_sqrt = alphas_bar_sqrt.cuda()
#之后公式中要用的
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_log = one_minus_alphas_bar_log.cuda()
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.cuda()
# 大小都一样，常数不需要训练
assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
==one_minus_alphas_bar_sqrt.shape
# print("all the same shape",betas.shape)

#给定初始，算出任意时刻采样值——正向扩散（前向过程）
# 计算任意时刻的x采样值，基于x_0和重参数化
def q_x(x_0, t):
    """可以基于x[0]得到任意时刻t的x[t]"""
    #生成正态分布采样
    noise = torch.randn_like(x_0)
    #得到均值方差
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    #根据x0求xt
    return (alphas_t * x_0 + alphas_1_m_t * noise)  # 在x[0]的基础上添加噪声


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(987655)
attention_heads = 4
attention_hidden = 64

learning_rate = 0.001
Epochs = 6
BATCH_SIZE = 16

T_stride = 2
T_overlop = T_stride / 2
overlapTime = {
    'neu': 1,
    'hap': 1,
    'sad': 1,
    'ang': 1,
}
FEATURES_TO_USE = 'mfcc'
impro_or_script = 'impro'
featuresFileName = 'E:/pytorch_project/speech_emotion_recognition/taslp/features4_{}_{}_CASIA.pkl'.format(FEATURES_TO_USE, impro_or_script)
# featuresFileName12 = './taslp/features_{}_{}_12channel.pkl'.format(FEATURES_TO_USE, impro_or_script)
# featuresFileName13 = './taslp/features_{}_{}_13channel.pkl'.format(FEATURES_TO_USE, impro_or_script)
WAV_PATH =  "E:/pytorch_project/speech_emotion_recognition/interspeech2023/interspeech21_emotion/path_to_wavs"
RATE = 16000
tripletData = 'E:/pytorch_project/speech_emotion_recognition/taslp/triplet3_3channel_CASIA.pkl'
# tripletData12 = 'triplet_12channel.pkl'
# tripletData13 = 'triplet_13channel.pkl'

dict = {
    'neu': torch.Tensor([0]),
    'hap': torch.Tensor([1]),
    'sad': torch.Tensor([2]),
    'ang': torch.Tensor([3]),
}
label_num = {
    'neu': 0,
    'hap': 0,
    'sad': 0,
    'ang': 0,
}


def plot_confusion_matrix(actual, predicted, labels, epochs_data , cmap=plt.cm.Blues):
    title='第{}个Epoch的混淆矩阵'.format(epochs_data+1)
    predicted = confusion_matrix(actual, predicted)
    # predicted = confusion_matrix(actual, predicted, labels)
    cm = predicted.astype('float') / predicted.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    print("confusion_matrix:",cm)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(labels) - 0.5, -0.5)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j] * 100, fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    os.makedirs("image", exist_ok=True)  # 创建目录，如果目录已存在则不报错
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在的绝对路径
    save_path = os.path.join(script_dir, "image", "pit_{}.jpg".format(epochs_data + 1))
    plt.savefig(save_path)
    # plt.show()


if __name__ == '__main__':

    with open(featuresFileName, 'rb')as f:
        features = pickle.load(f)
    train_X_features = features['train_X']
    train_y = features['train_y']  # ang: 704 neu: 2290 hap: 1271 sad: 1592
                                   # ang: 2835 neu: 3682 hap: 4302 sad: 3546
    valid_features_dict = features['val_dict']



    with open(tripletData, 'rb')as f:
        tripletlossData = pickle.load(f)
    anchor = tripletlossData['anchor']
    positive = tripletlossData['positive']
    negative = tripletlossData['negative']
    anchor_label = tripletlossData['anchor_label']
    pos_label = tripletlossData['pos_label']
    neg_label = tripletlossData['neg_label']


    with open('fake_label.pkl', 'rb')as f:
        fake_label = pickle.load(f)

    train_data = data_loader.TripletDataSet_fake(anchor, positive, negative, anchor_label, pos_label, neg_label,
                                                 fake_label)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # 扩散迭代周期
    num_epoch = 4000  # 4000
    # 扩散步长
    num_steps = 100
    # 实例化模型，传入一个数
    model = MLPDiffusion(num_steps).cuda()  # 输出维度是2，输入是x和step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    F_net = Encoder_Dm_SER3(attention_heads,attention_hidden)  # Encoder1为修改attention后的编码器，Encoder为原始版本，Encoder2为适应扩散模型修改的编码器
    # D_net = net.Discriminator1()
    # G_net = net.Generator1()
    # diffusion_block = GaussianDiffusionBlock(model_unet, image_size, timesteps)
    # G_net = Generator_Dm(model_unet, image_size, timesteps)
    D_net = Discriminator_Dm_SER3()
    Classifiar = Classifiar_Dm_SER3()
    fddpm = MLPDiffusion(num_steps).cuda()


    # TripletLoss = net.TripletLoss()
    if torch.cuda.is_available():
        F_net = F_net.cuda()
        D_net = D_net.cuda()
        # G_net = G_net.cuda()
        Classifiar = Classifiar.cuda()

        # TripletLoss = TripletLoss.cuda()

    softmax_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCEWithLogitsLoss()
    triplet_criterion = nn.TripletMarginLoss()

    optimizer_F = optim.Adam(F_net.parameters(), lr=learning_rate,
                             weight_decay=1e-6)
    optimizer_D = optim.Adam(D_net.parameters(), lr=learning_rate,
                             weight_decay=1e-6)
    # optimizer_G = optim.Adam(G_net.parameters(), lr=learning_rate,
    #                          weight_decay=1e-6)
    optimizer_classifiar = optim.Adam(Classifiar.parameters(), lr=learning_rate,
                                      weight_decay=1e-6)
    logging.info("training...")
    maxWA = 0
    maxUA = 0

    final_labels = ['NEU', 'HAP', 'SAD', 'ANG']
    label_to_class = {0: 'ANG', 1: 'SAD', 2: 'NEU', 3: 'HAP'}

    totalrunningTime = 0
    loss_all = []
    Acc_all = []
    UA_all = []
    UA_mean_all = []
    eporch_labels = []  # 保存学的最好的轮次

    for i in range(Epochs):
        optimizer_F.step()
        optimizer_D.step()
        optimizer_classifiar.step()
        epochs_data = i
        pred_model = []
        actual = []
        startTime = time.perf_counter()
        tq = tqdm(total=len(anchor_label))
        F_net.train()
        D_net.train()
        Classifiar.train()
        print_loss = 0
        for _, data in enumerate(train_loader):
            print_loss = 0
            x_anchor, x_pos, x_neg, y_pos, y_neg, fake_label = data
            if torch.cuda.is_available():
                x_anchor = x_anchor.cuda()
                x_pos = x_pos.cuda()
                x_neg = x_neg.cuda()
                y_pos = y_pos.cuda()
                y_neg = y_neg.cuda()
                fake = torch.zeros((y_pos.shape[0], 1)).cuda()
                real = torch.ones((y_pos.shape[0], 1)).cuda()
            #三元组特征提取部分
            F_anchor = F_net(x_anchor)
            F_positive = F_net(x_pos)
            F_negative = F_net(x_neg)
            # 拟合逆扩散过程高斯分布模型——拟合逆扩散时的噪声
            batch_size = BATCH_SIZE*3
            FX = torch.cat((F_anchor, F_positive, F_negative))
            # dataset放到dataloader中
            # dataset = torch.Tensor(FX).float().cuda()
            dataset = FX.float().to(device)  # 将CUDA张量移动到设备上并转换为浮点型
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            reverse_anchor = []
            reverse_positive = []
            reverse_negative = []
            for t in range(num_epoch):
                loss=0
                for idx, batch_x in enumerate(dataloader):
                    # 得到loss
                    loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    # 梯度clip，保持稳定性
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    if (t % 100 == 0):  # 并不是4000千个周期的逆向过程均使用，选择其中的40个逆向过程周期即可
                        print("扩散模型初始化训练损失", loss)
                        x_seq = p_sample_loop(model, dataset.shape, num_steps, betas,
                                              one_minus_alphas_bar_sqrt)  # 逆向过程的输出 是一个长度为t的列表
                        reverse_anchor.append(x_seq[-1][0:BATCH_SIZE,:].detach())  # 选择逆向过程的最后一个特征
                        reverse_positive.append(x_seq[-1][BATCH_SIZE:2*BATCH_SIZE,:].detach())  # 选择逆向过程的最后一个特征
                        reverse_negative.append(x_seq[-1][2*BATCH_SIZE:3*BATCH_SIZE,:].detach())  # 选择逆向过程的最后一个特征
            print("逆向过程数组长度：", len(reverse_anchor))
            #扩散模型部分的生成器输出
            FS_anchor = F_anchor.cuda()  # (16,256)
            FS_positive = F_positive.cuda()  # (16,256)
            FS_negative = F_negative.cuda()  # (16,256)
            num_epochs = len(reverse_anchor) #40
            print("开始交替训练")
            # 交替训练
            # for x in range(num_epochs):
            #     # D -> C  扩散模型到分类器
            #     # G_F_anchor = reverse_anchor[-1].cuda()
            #     # G_F_positive = reverse_positive[-1].cuda()
            #     # G_F_negative = reverse_negative[-1].cuda()
            #
            #     G_F_anchor = reverse_anchor[x].cuda()
            #     G_F_positive = reverse_positive[x].cuda()
            #     G_F_negative = reverse_negative[x].cuda()
            #
            #     D_G_F_anchor, D_G_F_anchor4 = D_net(
            #         G_F_anchor)
            #     D_G_F_positive, D_G_F_positive4 = D_net(
            #         G_F_positive)
            #     D_G_F_negative, D_G_F_negative4 = D_net(G_F_negative)
            #
            #     D_F_anchor, D_F_anchor4 = D_net(F_anchor)
            #     D_F_positive, D_F_positive4 = D_net(F_positive)
            #     D_F_negative, D_F_negative4 = D_net(F_negative)
            #
            #
            #     # C -> D  分类器到扩散模型
            #     # # 训练分类器
            #     G_F_anchor = G_F_anchor.detach()
            #     G_F_positive = G_F_positive.detach()
            #     G_F_negative = G_F_negative.detach()
            #     FS_anchor = F_anchor.detach() # (16,256)
            #     FS_positive = F_positive.detach()  # (16,256)
            #     FS_negative = F_negative.detach()  # (16,256)
            #
            #     anchor_out = Classifiar(torch.cat((FS_anchor, G_F_anchor)).cuda())
            #     positive_out = Classifiar(torch.cat((FS_positive, G_F_positive)).cuda())
            #     negative_out = Classifiar(torch.cat((FS_negative, G_F_negative)).cuda())
            #
            #     F_tri_loss = triplet_criterion(G_F_anchor, G_F_positive, G_F_negative)
            #     G_tri_loss = triplet_criterion(G_F_anchor, G_F_negative, G_F_positive)
            #
            #     DREALLOSS = (bce_criterion(D_F_anchor.squeeze(1), real.squeeze(1)) +
            #                  bce_criterion(D_F_positive.squeeze(1),real.squeeze(1)) +
            #                  bce_criterion(D_F_negative.squeeze(1),real.squeeze(1))) / 3
            #
            #     DREALLOSS1 = (softmax_criterion(D_F_anchor4.squeeze(1), y_pos.squeeze(1)) +
            #                   softmax_criterion(D_F_positive4.squeeze(1),y_pos.squeeze(1)) +
            #                   softmax_criterion(D_F_negative4.squeeze(1),y_neg.squeeze(1))) / 3
            #
            #     DFAKELOSS = (bce_criterion(D_G_F_anchor.squeeze(1), fake.squeeze(1)) +
            #                  bce_criterion(D_G_F_positive.squeeze(1),fake.squeeze(1)) +
            #                  bce_criterion(D_G_F_negative.squeeze(1),fake.squeeze(1))) / 3
            #
            #     GCLSLOSS = (softmax_criterion(D_G_F_anchor4, y_pos.squeeze(1)) +
            #                 softmax_criterion(D_G_F_positive4,y_pos.squeeze(1))+
            #                 softmax_criterion(D_G_F_negative4,y_neg.squeeze(1))) / 3
            #
            #     y_pos_anchor = torch.cat((y_pos.squeeze(1),y_pos.squeeze(1))).cuda()
            #     y_pos_positive = torch.cat((y_pos.squeeze(1),y_pos.squeeze(1))).cuda()
            #     y_neg_negative = torch.cat((y_neg.squeeze(1),y_neg.squeeze(1))).cuda()
            #
            #     Classifiar_loss = (softmax_criterion(anchor_out, y_pos_anchor) +
            #                        softmax_criterion(positive_out,y_pos_positive) +
            #                        softmax_criterion(negative_out,y_neg_negative)) / 3  #分类网咯损失
            #
            #     F_loss = F_tri_loss + Classifiar_loss  # encode 主干网络
            #     D_loss = DREALLOSS + DFAKELOSS + DREALLOSS1 # discriminator 辨别器
            #     classlabel = Classifiar_loss + GCLSLOSS  # 分类器损失
            #
            #     loss = classlabel + F_loss + D_loss
            #     print_loss += loss.data.item() * BATCH_SIZE
            #
            #     optimizer_F.zero_grad()
            #     optimizer_D.zero_grad()
            #     optimizer_classifiar.zero_grad()
            #
            #
            #     F_loss.backward(retain_graph=True)
            #     D_loss.backward(retain_graph=True)
            #     classlabel.backward(retain_graph=True)

            # 交替训练
            # for x in range(num_epochs):
            # D -> C  扩散模型到分类器
            G_F_anchor = reverse_anchor[-1].cuda()
            G_F_positive = reverse_positive[-1].cuda()
            G_F_negative = reverse_negative[-1].cuda()

            D_G_F_anchor, D_G_F_anchor4 = D_net(G_F_anchor)
            D_G_F_positive, D_G_F_positive4 = D_net(G_F_positive)
            D_G_F_negative, D_G_F_negative4 = D_net(G_F_negative)

            D_F_anchor, D_F_anchor4 = D_net(F_anchor)
            D_F_positive, D_F_positive4 = D_net(F_positive)
            D_F_negative, D_F_negative4 = D_net(F_negative)

            # C -> D  分类器到扩散模型
            # # 训练分类器
            G_F_anchor = G_F_anchor.detach()
            G_F_positive = G_F_positive.detach()
            G_F_negative = G_F_negative.detach()
            FS_anchor = F_anchor.detach()  # (16,256)
            FS_positive = F_positive.detach()  # (16,256)
            FS_negative = F_negative.detach()  # (16,256)

            anchor_out = Classifiar(torch.cat((FS_anchor, G_F_anchor)).cuda())
            positive_out = Classifiar(torch.cat((FS_positive, G_F_positive)).cuda())
            negative_out = Classifiar(torch.cat((FS_negative, G_F_negative)).cuda())

            F_tri_loss = triplet_criterion(G_F_anchor, G_F_positive, G_F_negative)
            G_tri_loss = triplet_criterion(G_F_anchor, G_F_negative, G_F_positive)

            DREALLOSS = (bce_criterion(D_F_anchor.squeeze(1), real.squeeze(1)) +
                         bce_criterion(D_F_positive.squeeze(1), real.squeeze(1)) +
                         bce_criterion(D_F_negative.squeeze(1), real.squeeze(1))) / 3

            DREALLOSS1 = (softmax_criterion(D_F_anchor4.squeeze(1), y_pos.squeeze(1)) +
                          softmax_criterion(D_F_positive4.squeeze(1), y_pos.squeeze(1)) +
                          softmax_criterion(D_F_negative4.squeeze(1), y_neg.squeeze(1))) / 3

            DFAKELOSS = (bce_criterion(D_G_F_anchor.squeeze(1), fake.squeeze(1)) +
                         bce_criterion(D_G_F_positive.squeeze(1), fake.squeeze(1)) +
                         bce_criterion(D_G_F_negative.squeeze(1), fake.squeeze(1))) / 3

            GCLSLOSS = (softmax_criterion(D_G_F_anchor4, y_pos.squeeze(1)) +
                        softmax_criterion(D_G_F_positive4, y_pos.squeeze(1)) +
                        softmax_criterion(D_G_F_negative4, y_neg.squeeze(1))) / 3

            y_pos_anchor = torch.cat((y_pos.squeeze(1), y_pos.squeeze(1))).cuda()
            y_pos_positive = torch.cat((y_pos.squeeze(1), y_pos.squeeze(1))).cuda()
            y_neg_negative = torch.cat((y_neg.squeeze(1), y_neg.squeeze(1))).cuda()

            Classifiar_loss = (softmax_criterion(anchor_out, y_pos_anchor) +
                               softmax_criterion(positive_out, y_pos_positive) +
                               softmax_criterion(negative_out, y_neg_negative)) / 3  # 分类网咯损失

            F_loss = F_tri_loss + Classifiar_loss  # encode 主干网络
            D_loss = DREALLOSS + DFAKELOSS + DREALLOSS1  # discriminator 辨别器
            classlabel = Classifiar_loss + GCLSLOSS  # 分类器损失

            loss = classlabel + F_loss + D_loss
            print_loss += loss.data.item()
            print("当前轮次损失为：", print_loss)

            optimizer_F.zero_grad()
            optimizer_D.zero_grad()
            optimizer_classifiar.zero_grad()

            F_loss.backward(retain_graph=True)
            D_loss.backward(retain_graph=True)
            classlabel.backward(retain_graph=True)



            tq.update(BATCH_SIZE)
        tq.close()
        print('epoch: {}, loss: {:.4f}'.format(i, print_loss / (len(train_X_features) * 40)))
        logging.info('epoch: {}, loss: {:.4f}'.format(i, print_loss / (len(train_X_features) * 40)))
        loss_all.append(print_loss / (len(train_X_features) * 40))
        if (i > 0 and i % 10 == 0):
            learning_rate = learning_rate / 10
            for param_group in optimizer_F.param_groups:
                param_group['lr'] = learning_rate
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = learning_rate
            # for param_group in optimizer_G.param_groups:
            #     param_group['lr'] = learning_rate
            for param_group in optimizer_classifiar.param_groups:
                param_group['lr'] = learning_rate
        # validation
        endTime = time.perf_counter()
        # endTime = time.clock()
        totalrunningTime += endTime - startTime
        print("totalrunningTime：",totalrunningTime)

        F_net.eval()
        D_net.eval()
        # G_net.eval()
        Classifiar.eval()
        UA = [0, 0, 0, 0]
        num_correct = 0
        class_total = [0, 0, 0, 0]
        matrix = np.mat(np.zeros((4, 4)), dtype=int)
        for _, i in enumerate(valid_features_dict):
            x, y = valid_features_dict[i]['X'], valid_features_dict[i]['y']
            x = torch.from_numpy(x).float()
            y = dict[y[0]].long()
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            # if (x.shape[0] != 16):
            #     x=x.permute(1,0,2,3)
            #     conv_two = nn.Conv2d(kernel_size=(1, 1), in_channels=x.shape[1], out_channels=16)
            #     conv_two.cuda()
            #     x=conv_two(x)
            #     x = x.reshape(-1,3,26,63)


            # if x.shape[0] == 16:
            out = F_net(x)
            out = Classifiar(out)
            pred = torch.Tensor([0, 0, 0, 0]).cuda()
            for j in range(out.size(0)):
                pred += out[j]
            pred = pred / out.size(0)
            pred = torch.max(pred, 0)[1]

            pred_model.append(pred.cpu().numpy())
            actual.append(y.data.cpu().numpy())
            # else:
            #     print("断开了")
            #     continue

            if (pred == y):
                num_correct += 1
            matrix[int(y), int(pred)] += 1

        for i in range(4):
            for j in range(4):
                class_total[i] += matrix[i, j]
            UA[i] = round(matrix[i, i] / class_total[i], 3)
        WA = num_correct / len(valid_features_dict)
        if (maxWA < WA):
            maxWA = WA
            torch.save(F_net, r'net.pkl')  # 保存整个神经网络到net1.pkl中
            torch.save(F_net.state_dict(), r'best_model_weights.pth')  # 保存网络里的参数
            eporch_labels.append(epochs_data)

        if (maxUA < sum(UA) / 4):
            maxUA = sum(UA) / 4

        print('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA:{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
        Acc_all.append(WA)
        UA_all.append(UA)
        UA_mean_all.append(sum(UA) / 4)
        logging.info('Acc: {:.6f}\nUA:{},{}\nmaxWA:{},maxUA:{}'.format(WA, UA, sum(UA) / 4, maxWA, maxUA))
        print(matrix)
        logging.info(matrix)
        pred_model = tuple(pred_model)
        pred_model = np.hstack(pred_model)
        actual = tuple(actual)
        actual = np.hstack(actual)
        pred_with_label = [label_to_class[label] for label in list(pred_model)]
        actual_with_label = [label_to_class[label] for label in list(actual)]
        plot_confusion_matrix(actual_with_label, pred_with_label, final_labels,epochs_data)
        print('\n Classification Report \n {} \n'.format(classification_report(actual_with_label, pred_with_label)))
    # torch.save(model, r'net.pkl')  # 保存整个神经网络到net1.pkl中
    # torch.save(model.state_dict(), r'net_params.pkl')  # 保存网络里的参数
    # 创建一个字典，将列表作为值，键作为列名
    data = {
        'loss': loss_all,
        'accuracy': Acc_all,
        'UA': UA_all,
        'UA_mean': UA_mean_all
    }
    df = pd.DataFrame(data)  # 使用字典创建数据框
    df.to_excel('data.xlsx', index=False)  # 保存训练数据至excel中
    print("模型训练的最好轮次为：", eporch_labels[-1])
