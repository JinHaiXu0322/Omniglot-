import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import scipy.io as scio
from torchvision import transforms
import random
from myutils.utils import preprocess_input
from scipy.ndimage import rotate

def horizontal_flip(img):
    if random.random() < 0.5:
        img = np.fliplr(img)
    return img


def vertical_flip(img):
    if random.random() < 0.5:
        img = np.flipud(img)
    return img


def random_crop(img):
    h, w = img.shape[:2]
    new_h, new_w = int(h * 0.9), int(w * 0.9)
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)
    img = img[top: top + new_h, left: left + new_w]
    return img

def random_rotation(img):
    angle = random.uniform(-10, 10)
    img = rotate(img, angle, reshape=False, mode='nearest')
    return img

class Dataset(Dataset):
    def __init__(self, path, input_shape, epoch_length, is_train):
        super(Dataset, self).__init__()
        self.path = path
        self.input_shape = input_shape
        self.epoch_length = epoch_length

        self.epoch_now = -1
        self.data = scio.loadmat(path)
        self.is_train = is_train

        if is_train:
            self.length = np.shape(self.data['train'])[0] * 15
            self.data = self.data['train']
        else:
            self.length = np.shape(self.data['test'])[0] * 5
            self.data = self.data['test']

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.is_train:
            index = index % self.length
            label = index // 15
            index = index % 15
        else:
            index = index % self.length
            label = index // 5
            index = index % 5
        # print(label, index)
        image = self.data[label][index]
        image = np.expand_dims(image, axis=2)

        if self.is_train:
            image = horizontal_flip(image)
            image = vertical_flip(image)
            image = random_crop(image)
            # image = color_jitter(image)
            image = random_rotation(image)
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))

        return image, label



# DataLoader中collate_fn使用
def dataset_collate(batch):
    images  = []
    Label = []
    for img, label in batch:
        images.append(img)
        Label.append(label)

    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    Label   = torch.from_numpy(np.array(Label)).type(torch.LongTensor)
    return images, Label
