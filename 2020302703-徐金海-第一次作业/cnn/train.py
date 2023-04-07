import os.path   # 导入os.path模块，用于操作文件路径
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
import torch    # 导入PyTorch库
import torch.backends.cudnn as cudnn    # 导入cudnn模块，用于提升PyTorch的计算速度
from torch.utils.data import DataLoader # 导入DataLoader类，用于加载数据集
from myutils.dataloader_cl import Dataset, dataset_collate    # 导入自定义数据集和数据集处理函数
from myutils.trainer import fit_one_epoch     # 导入自定义训练函数
from nets.model import Baseline     # 导入自定义神经网络模型

if __name__ == "__main__":
    Cuda = False    # 是否使用GPU进行训练，默认为False
    pretrained_model_path  = 'logs/last_epoch_weights.pth'# 预训练模型的路径，默认为空logs/last_epoch_weights.pth
    input_shape = [28, 28]  # 输入图像的大小，默认为28*28
    batch_size = 64 # 每个batch的大小，默认为32
    Init_Epoch = 0      # 起始迭代次数，默认为0
    Epoch = 300     # 起始迭代次数，默认为0
    lr = 0.0003
    save_period = 50        # 每隔多少个epoch保存一次模型权重，默认为50
    save_dir = 'logs/'   # 权重和日志文件保存的文件夹名称，默认为'logs/'
    # 如果指定文件夹不存在，则创建文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    num_workers = 0     # 数据集加载器使用的进程数，默认为0
    train_val_dataset_path = 'dataset/data.mat'
    ngpus_per_node = torch.cuda.device_count()  # 获取可用的GPU数量
    # 设置设备为GPU，如果没有GPU则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Baseline()  # 实例化模型对象
    if pretrained_model_path != '':  # 如果预训练模型路径不为空，则加载预训练模型的权重
        print('Load weights {}.'.format(pretrained_model_path))
        pretrained_dict = torch.load(pretrained_model_path, map_location= device)
        model.load_state_dict(pretrained_dict)
    model_train = model.train()     # 将模型设置为训练模式

    # 如果使用 GPU 进行训练，则使用 DataParallel 将模型转换为可以在多个 GPU 上运行的模型，并设置 cudnn.benchmark 以加速模型的计算
    if Cuda:
        Generator_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        Generator_train = Generator_train.cuda()

    opt_model = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    # 构建学习率衰减，以下是两种，在动态调整模型参数时使用
    # scheduler = StepLR(opt_model, step_size=1000, gamma=0.1)
    # lr_scheduler = MultiStepLR(opt_model, milestones= [20, 80, 160, 240], gamma=0.1, last_epoch=-1, verbose=False)
    train_dataset = Dataset(train_val_dataset_path, input_shape, epoch_length=Epoch, is_train=True)
    val_dataset = Dataset(train_val_dataset_path, input_shape, epoch_length=Epoch, is_train=False)
    shuffle = True  # 表示是否打乱数据集

    # 创建训练集数据加载器，使用 PyTorch 内置的 DataLoader 类，并将训练集数据集、是否打乱数据集、batch_size、线程数、是否将数据加载到 GPU 上、是否丢弃最后一批数据、数据集合并方式、采样器等作为参数
    train_gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=dataset_collate, sampler=None)
    # 创建测试集数据加载器
    val_gen = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=dataset_collate, sampler=None)

    for epoch in range(Init_Epoch, Epoch):
        # 计算每个 epoch 中的步数
        epoch_step = train_dataset.length // batch_size
        epoch_step_val = val_dataset.length // batch_size
        # 更新数据集的当前 epoch 值
        train_gen.dataset.epoch_now = epoch
        val_gen.dataset.epoch_now = epoch
        # # 使用 fit_one_epoch 函数进行模型训练
        fit_one_epoch(model_train, model, opt_model, epoch, epoch_step, epoch_step_val, train_gen, val_gen, Epoch, Cuda, save_period, save_dir)
        # scheduler.step()
        # lr_scheduler.step()
        opt_model.step()

