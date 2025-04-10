import pickle
import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split

FEATURES_TO_USE = 'mfcc'
impro_or_script = 'impro'
# tripletData = 'triplet_3channel.pkl'
tripletData = 'triplet_3channel_EMODB_7emo.pkl'
featuresFileName = 'features_{}_{}_EMODB_7emo.pkl'.format(FEATURES_TO_USE, impro_or_script)
# featuresFileName = 'features_{}_{}_3channel.pkl'.format(FEATURES_TO_USE, impro_or_script)
with open(featuresFileName, 'rb') as f:
    features = pickle.load(f)
train_X_features = features['train_X']  # ang: 704 neu: 2290 hap: 1271 sad: 1592  total:5857
train_y = features['train_y']
valid_features_dict = features['val_dict']

hap, sad, neu, ang, fea, dis,bored = [], [], [], [], [], [],[]  # ang: 700 neu: 2290 hap: 1270 sad: 1590  total:5850
a, b, c, d, j, k, l, n = 0, 0, 0, 0, 0, 0, 0,0
for i in train_y:
    # if i == 'happy':
    if i == 'hap':
        hap.append(train_X_features[n])
        a = a + 1
    if i == 'sad':
        sad.append(train_X_features[n])
        b = b + 1
    # if i == 'neutral':
    if i == 'neu':
        neu.append(train_X_features[n])
        c = c + 1
    # if i == 'angry':
    if i == 'ang':
        ang.append(train_X_features[n])
        d = d + 1
    if i == 'fea':
        fea.append(train_X_features[n])
        j = j + 1
    if i == 'ang':
        dis.append(train_X_features[n])
        k = k + 1
    if i == 'bored':
        bored.append(train_X_features[n])
        l = l + 1
    n = n + 1

anchor, positive, negative = [], [], []
anchor_label, pos_label, neg_label = [], [], []

# for k in range(4000):
#     random.shuffle(neu)
#     anchor.append(neu[0])
#     positive.append(neu[1])
#     anchor_label.append('neutral')
#     pos_label.append('neutral')
#     random.shuffle(hap)
#     negative.append(hap[0])
#     neg_label.append('happy')
#
# for j in range(6000):
#     anchor_id = 2
#     random.shuffle(neu)
#     anchor.append(neu[0])
#     positive.append(neu[1])
#     anchor_label.append('neutral')
#     pos_label.append('neutral')
#     List = [1, 3]
#     Lst1 = random.choice(List)
#     if Lst1 == 1:
#         random.shuffle(sad)
#         negative.append(sad[0])
#         neg_label.append('sad')
#     if Lst1 == 3:
#         random.shuffle(ang)
#         negative.append(ang[0])
#         neg_label.append('angry')

for i in range(50000):
    print(i)
    anchor_id = random.randint(0, 7)
    if anchor_id == 0:
        random.shuffle(hap)
        anchor.append(hap[0])
        positive.append(hap[1])
        anchor_label.append('happy')
        pos_label.append('happy')
        List = [1, 2, 3, 4, 5,6]
        Lst1 = random.choice(List)
        if Lst1 == 1:
            random.shuffle(sad)
            negative.append(sad[0])
            neg_label.append('sad')
        if Lst1 == 2:
            random.shuffle(neu)
            negative.append(neu[0])
            neg_label.append('neutral')
        if Lst1 == 3:
            random.shuffle(ang)
            negative.append(ang[0])
            neg_label.append('angry')
        if Lst1 == 4:
            random.shuffle(fea)
            negative.append(ang[0])
            neg_label.append('fearful')
        if Lst1 == 5:
            random.shuffle(dis)
            negative.append(ang[0])
            neg_label.append('dis')
        if Lst1 == 6:
            random.shuffle(bored)
            negative.append(ang[0])
            neg_label.append('bored')
    if anchor_id == 1:
        random.shuffle(sad)
        anchor.append(sad[0])
        positive.append(sad[1])
        anchor_label.append('sad')
        pos_label.append('sad')
        List = [0, 2, 3, 4, 5,6]
        Lst1 = random.choice(List)
        if Lst1 == 0:
            random.shuffle(hap)
            negative.append(hap[0])
            neg_label.append('happy')
        if Lst1 == 2:
            random.shuffle(neu)
            negative.append(neu[0])
            neg_label.append('neutral')
        if Lst1 == 3:
            random.shuffle(ang)
            negative.append(ang[0])
            neg_label.append('angry')
        if Lst1 == 4:
            random.shuffle(fea)
            negative.append(ang[0])
            neg_label.append('fearful')
        if Lst1 == 5:
            random.shuffle(dis)
            negative.append(ang[0])
            neg_label.append('dis')
        if Lst1 == 6:
            random.shuffle(bored)
            negative.append(ang[0])
            neg_label.append('bored')
    if anchor_id == 2:
        random.shuffle(neu)
        anchor.append(neu[0])
        positive.append(neu[1])
        anchor_label.append('neutral')
        pos_label.append('neutral')
        List = [1, 0, 3, 4, 5,6]
        Lst1 = random.choice(List)
        if Lst1 == 0:
            random.shuffle(hap)
            negative.append(hap[0])
            neg_label.append('happy')
        if Lst1 == 1:
            random.shuffle(sad)
            negative.append(sad[0])
            neg_label.append('sad')
        if Lst1 == 3:
            random.shuffle(ang)
            negative.append(ang[0])
            neg_label.append('angry')
        if Lst1 == 4:
            random.shuffle(fea)
            negative.append(ang[0])
            neg_label.append('fearful')
        if Lst1 == 5:
            random.shuffle(dis)
            negative.append(ang[0])
            neg_label.append('dis')
        if Lst1 == 6:
            random.shuffle(bored)
            negative.append(ang[0])
            neg_label.append('bored')
    if anchor_id == 3:
        random.shuffle(ang)
        anchor.append(ang[0])
        positive.append(ang[1])
        anchor_label.append('angry')
        pos_label.append('angry')
        List = [1, 2, 0, 4, 5,6]
        Lst1 = random.choice(List)
        if Lst1 == 0:
            random.shuffle(hap)
            negative.append(hap[0])
            neg_label.append('happy')
        if Lst1 == 1:
            random.shuffle(sad)
            negative.append(sad[0])
            neg_label.append('sad')
        if Lst1 == 2:
            random.shuffle(neu)
            negative.append(neu[0])
            neg_label.append('neutral')
        if Lst1 == 4:
            random.shuffle(fea)
            negative.append(ang[0])
            neg_label.append('fearful')
        if Lst1 == 5:
            random.shuffle(dis)
            negative.append(ang[0])
            neg_label.append('dis')
        if Lst1 == 6:
            random.shuffle(bored)
            negative.append(ang[0])
            neg_label.append('bored')
    if anchor_id == 4:
        random.shuffle(fea)
        anchor.append(fea[0])
        positive.append(fea[1])
        anchor_label.append('fearful')
        pos_label.append('fearful')
        List = [1, 2, 3, 0, 5, 6]
        Lst1 = random.choice(List)
        if Lst1 == 0:
            random.shuffle(hap)
            negative.append(hap[0])
            neg_label.append('happy')
        if Lst1 == 1:
            random.shuffle(sad)
            negative.append(sad[0])
            neg_label.append('sad')
        if Lst1 == 2:
            random.shuffle(neu)
            negative.append(neu[0])
            neg_label.append('neutral')
        if Lst1 == 3:
            random.shuffle(ang)
            negative.append(ang[0])
            neg_label.append('angry')
        if Lst1 == 5:
            random.shuffle(dis)
            negative.append(ang[0])
            neg_label.append('dis')
        if Lst1 == 6:
            random.shuffle(bored)
            negative.append(ang[0])
            neg_label.append('bored')
    if anchor_id == 5:
        random.shuffle(dis)
        anchor.append(dis[0])
        positive.append(dis[1])
        anchor_label.append('dis')
        pos_label.append('dis')
        List = [1, 2, 3, 4, 0, 6]
        Lst1 = random.choice(List)
        if Lst1 == 0:
            random.shuffle(hap)
            negative.append(hap[0])
            neg_label.append('happy')
        if Lst1 == 1:
            random.shuffle(sad)
            negative.append(sad[0])
            neg_label.append('sad')
        if Lst1 == 2:
            random.shuffle(neu)
            negative.append(neu[0])
            neg_label.append('neutral')
        if Lst1 == 3:
            random.shuffle(ang)
            negative.append(ang[0])
            neg_label.append('angry')
        if Lst1 == 4:
            random.shuffle(fea)
            negative.append(ang[0])
            neg_label.append('fearful')
        if Lst1 == 6:
            random.shuffle(bored)
            negative.append(ang[0])
            neg_label.append('bored')
    if anchor_id == 6:
        random.shuffle(bored)
        anchor.append(bored[0])
        positive.append(bored[1])
        anchor_label.append('bored')
        pos_label.append('bored')
        List = [1, 2, 3, 4, 5, 0]
        Lst1 = random.choice(List)
        if Lst1 == 0:
            random.shuffle(hap)
            negative.append(hap[0])
            neg_label.append('happy')
        if Lst1 == 1:
            random.shuffle(sad)
            negative.append(sad[0])
            neg_label.append('sad')
        if Lst1 == 2:
            random.shuffle(neu)
            negative.append(neu[0])
            neg_label.append('neutral')
        if Lst1 == 3:
            random.shuffle(ang)
            negative.append(ang[0])
            neg_label.append('angry')
        if Lst1 == 4:
            random.shuffle(fea)
            negative.append(ang[0])
            neg_label.append('fearful')
        if Lst1 == 5:
            random.shuffle(dis)
            negative.append(ang[0])
            neg_label.append('dis')


anchor = np.array(anchor)
positive = np.array(positive)
negative = np.array(negative)
anchor_label = np.array(anchor_label)
pos_label = np.array(pos_label)
neg_label = np.array(neg_label)
a, b, c, d, j, k, l = 0, 0, 0, 0, 0, 0, 0
for i in anchor_label:
    if i == 'happy':
        a = a + 1
    if i == 'sad':
        b = b + 1
    if i == 'neutral':
        c = c + 1
    if i == 'angry':
        d = d + 1
    if i == 'fearful':
        j = j + 1
    if i == 'dis':
        k = k + 1
    if i == 'bored':
        l = l + 1
print(a, b, c, d, j, k, l)
# happy:10125  sad:9878  neutral:19963  angry:10034
features = {'anchor': anchor, 'positive': positive, 'negative': negative,
            'anchor_label': anchor_label, 'pos_label': pos_label, 'neg_label': neg_label, }
with open(tripletData, 'wb') as f:
    pickle.dump(features, f)
