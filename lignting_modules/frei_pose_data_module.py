import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader


class FreiPoseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, dataset, train_ratio=0.9):
        super().__init__()
        train_dataset_size = int(len(dataset) * train_ratio)
        self.batch_size = batch_size
        self.train_dataset, self.val_dataset = random_split(dataset,
                                                            [train_dataset_size, len(dataset) - train_dataset_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
