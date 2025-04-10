import torch
from torch.utils.data import Dataset, DataLoader

dict = {
    'neutral': torch.Tensor([0]),
    'happy': torch.Tensor([1]),
    'sad': torch.Tensor([2]),
    'angry': torch.Tensor([3]),
}
dict_fake = {
    'neutral': torch.Tensor([0]),
    'happy': torch.Tensor([1]),
    'sad': torch.Tensor([2]),
    'angry': torch.Tensor([3]),
    'fake': torch.Tensor([4]),
}


class DataSet(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x = self.X[index]
        # x = torch.from_numpy(x).unsqueeze(0)
        x = torch.from_numpy(x)
        x = x.float()
        y = self.Y[index]
        y = dict[y]
        y = y.long()
        return x, y

    def __len__(self):
        return len(self.X)


class TripletDataSet(Dataset):
    # def __init__(self, anchor, positive, negative, anchor_label, pos_label, neg_label, fake_label):
    def __init__(self, anchor, positive, negative, anchor_label, pos_label, neg_label):
        self.X_anchor = anchor
        self.X_pos = positive
        self.X_neg = negative
        # self.Y_anchor = anchor_label
        self.Y_pos = pos_label
        self.Y_neg = neg_label
        # self.Fake_label = fake_label

    def __getitem__(self, index):
        x_anchor = self.X_anchor[index]
        x_pos = self.X_pos[index]
        x_neg = self.X_neg[index]
        x_anchor = torch.from_numpy(x_anchor)
        x_pos = torch.from_numpy(x_pos)
        x_neg = torch.from_numpy(x_neg)
        x_anchor = x_anchor.float()
        x_pos = x_pos.float()
        x_neg = x_neg.float()

        y_pos = self.Y_pos[index]
        y_neg = self.Y_neg[index]
        # fake_label = self.Fake_label[index]

        y_pos = dict[y_pos]
        y_neg = dict[y_neg]
        # fake_label = dict_fake[fake_label]
        # fake_label = fake_label.long()
        y_pos = y_pos.long()
        y_neg = y_neg.long()

        return x_anchor, x_pos, x_neg, y_pos, y_neg
        # return x_anchor, x_pos, x_neg, y_pos, y_neg, fake_label

    def __len__(self):
        return len(self.X_anchor)

class TripletDataSet_fake(Dataset):
    def __init__(self, anchor, positive, negative, anchor_label, pos_label, neg_label, fake_label):
        self.X_anchor = anchor
        self.X_pos = positive
        self.X_neg = negative
        # self.Y_anchor = anchor_label
        self.Y_pos = pos_label
        self.Y_neg = neg_label
        self.Fake_label = fake_label

    def __getitem__(self, index):
        x_anchor = self.X_anchor[index]
        x_pos = self.X_pos[index]
        x_neg = self.X_neg[index]
        x_anchor = torch.from_numpy(x_anchor)
        x_pos = torch.from_numpy(x_pos)
        x_neg = torch.from_numpy(x_neg)
        x_anchor = x_anchor.float()
        x_pos = x_pos.float()
        x_neg = x_neg.float()

        y_pos = self.Y_pos[index]
        y_neg = self.Y_neg[index]
        fake_label = self.Fake_label[index]

        y_pos = dict[y_pos]
        y_neg = dict[y_neg]
        fake_label = dict_fake[fake_label]
        fake_label = fake_label.long()
        y_pos = y_pos.long()
        y_neg = y_neg.long()

        return x_anchor, x_pos, x_neg, y_pos, y_neg, fake_label

    def __len__(self):
        return len(self.X_anchor)