import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# 加载.mat文件
data = scipy.io.loadmat('NewDataset.mat')

# 获取train和test数据
train_images = np.zeros((200, 15, 28, 28))
test_images = np.zeros((200, 5, 28, 28))

for i in range(200):
    train_images[i] = data['train'][i][:15].reshape(15, 28, 28)
    test_images[i] = data['test'][i][:5].reshape(5, 28, 28)

# 创建train和test文件夹
if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('test'):
    os.makedirs('test')

# 保存train图像
for i in range(200):
    combined_image = np.concatenate(train_images[i], axis=1)
    filename = f'train/{i:03d}_train.png' # 序号命名
    plt.imsave(filename, combined_image, cmap='gray')

# 保存test图像
for i in range(200):
    combined_image = np.concatenate(test_images[i], axis=1)
    filename = f'test/{i:03d}_test.png' # 序号命名
    plt.imsave(filename, combined_image, cmap='gray')
