import scipy.io as sio
import numpy as np

# 加载原始数据文件
raw_data = sio.loadmat('NewDataset.mat')

# 将原始数据中的所有行复制到新的变量中
new_train_data = np.copy(raw_data['train'])
new_test_data = np.copy(raw_data['test'])

# 删除第121和122行
new_train_data = np.delete(new_train_data, [120, 121], axis=0)
new_test_data = np.delete(new_test_data, [120, 121], axis=0)

# 保存处理后的数据到新的MATLAB格式文件
sio.savemat('data.mat', {'train': new_train_data, 'test': new_test_data})

