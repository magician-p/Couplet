import torch
from tqdm import tqdm

from config import *


def train_epoch(net, datalodar, optim, device, epoch):
    net.train()
    loss_ = 0
    for batch in tqdm(datalodar, desc=f'Epoch {epoch}/{EPOCHS} train: ', leave=False, position=1, ncols=100):
        inputs = {
            k: v.to(device) for k, v in batch.items()
        }

        loss = net(**inputs).loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_ += loss.item()
    return loss_/len(datalodar)

def val(net, datalodar, device, epoch):
    net.eval()
    loss_ = 0
    loss_num = 0
    with torch.no_grad():
        for batch in tqdm(datalodar, desc=f'Epoch {epoch}/{EPOCHS} val: ', leave=False, position=1, ncols=100):
            inputs = {
                k: v.to(device) for k, v in batch.items()
            }

            loss = net(**inputs).loss

            loss_ += loss.item() * inputs['input_ids'].shape[0]
            loss_num += inputs['input_ids'].shape[0]
        return loss_/loss_num