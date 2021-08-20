import os
from more_itertools import mark_ends
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl

def make_dataloaders(dataset, split_ratios = (9, 1), **dataloader_kwargs):
    N_total = len(dataset)
    ratio_total = sum(split_ratios)
    
    split_lengths = []
    for _, is_last, ratio in mark_ends(split_ratios):
        N_split = N_total - sum(split_lengths) if is_last else int(N_total*ratio/ratio_total)
        split_lengths.append(N_split)

    return [DataLoader(split_dataset, **dataloader_kwargs) for split_dataset in random_split(dataset, split_lengths)]


def make_trainer(savedir, **trainer_kwargs):
    log_dir = os.path.join(savedir, 'logs/')
    logger = pl.loggers.TensorBoardLogger(save_dir = log_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor = 'validation loss', save_top_k = 1, mode = 'min', every_n_epochs = 1, verbose = True) # need to use the every_n_epochs = 1 kwarg?

    trainer = pl.Trainer(logger = logger, callbacks = [checkpoint_callback], **trainer_kwargs)
    return trainer