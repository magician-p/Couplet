import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM

from config import *
from dataset import get_dataloader
from engine import train_epoch, val


def train():
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader = get_dataloader(
        TRAIN_DATA_PATH,
        batch_size=BATCH_SIZE,
        data_type='train',
        shuffle=True,
        drop_last=True
    )
    test_dataloader = get_dataloader(
        TEST_DATA_PATH,
        batch_size=BATCH_SIZE,
        data_type='test',
        shuffle=False,
        drop_last=False
    )

    net = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL_NAME)
    net.to(device)

    optim = torch.optim.AdamW(net.parameters(), lr=LR)

    min_loss = float('inf')
    for epoch in tqdm(range(EPOCHS), desc='Training', leave=True, position=0, ncols=100):
        train_loss = train_epoch(net, train_dataloader, optim, device, epoch)
        tqdm.write(f'Epoch {epoch} Train Loss: {train_loss:.4f}')

        val_loss = val(net, test_dataloader, device, epoch)
        tqdm.write(f'Val Epoch {epoch} Loss: {val_loss:.4f}')

        if val_loss < min_loss:
            min_loss = val_loss
            net.save_pretrained(str(MODEL_DIR))
            tqdm.write(f'Saved model to {MODEL_DIR}')


if __name__ == "__main__":
    train()