import os

import cv2
import kornia
import numpy
from torch import Tensor
import torch.nn as nn
import torch
from tqdm import tqdm


def fit_one_epoch(model_train, model, opt_model, epoch, epoch_step, epoch_step_val, train_gen, val_gen, Epoch,
                  cuda, save_period, save_dir):
    loss = 0
    train_set = set()
    print('Start Train')
    criterion = nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    acc = 0
    for iteration, batch in enumerate(train_gen):
        if iteration >= epoch_step:
            break

        images, label = batch[0], batch[1]  # image (B,C,H,W)   label (B)
        with torch.no_grad():
            if cuda:
                images = images.cuda()
                label = label.cuda()

        model_train.train()

        prob_tensor = model_train(images)
        class_index = torch.argmax(prob_tensor, dim=1)

        acc = acc + (label == class_index).sum().item()
        loss_value = criterion(prob_tensor, label)

        opt_model.zero_grad()
        loss_value.backward()
        opt_model.step()

        loss += loss_value.item()

        pbar.set_postfix(**{'loss': loss / (iteration + 1),
                            'acc': acc / ((iteration + 1) * label.shape[0])
                            })
        pbar.update(1)

    print('Start test')
    pbar.close()
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    acc = 0
    for iteration, batch in enumerate(val_gen):
        if iteration >= epoch_step_val:
            break

        model_train.eval()
        images, label = batch[0], batch[1]
        for i in range(label.shape[0]):
            train_set.add(int(label[i]))
        with torch.no_grad():
            if cuda:
                images = images.cuda()
                label = label.cuda()

        prob_tensor = model_train(images)
        class_index = torch.argmax(prob_tensor, dim=1)

        acc = acc + (label == class_index).sum().item()

        pbar.set_postfix(**{'acc': acc / ((iteration + 1) * label.shape[0]),
                            })
        pbar.update(1)
    pbar.close()

    save_state_dict = model.state_dict()

    # save_state_dict_gen = Generator.state_dict()

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f.pth" % (epoch + 1, loss / epoch_step)))

    torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))