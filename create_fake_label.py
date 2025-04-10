import torch
import numpy as np
import pickle

num = ['fake' for i in range(1, 50001)]  # 创建一维数组
print("-" * 50)  # 分割线

fake_label = np.array(num)
# fake_label = torch.from_numpy(fake_label)

with open('fake_label.pkl', 'wb') as f:
    pickle.dump(fake_label, f)

with open('fake_label.pkl', 'rb')as f:
    tripletlossData = pickle.load(f)

print("-" * 50)  # 分割线
