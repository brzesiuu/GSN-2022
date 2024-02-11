from torch.utils.data import Dataset


class SourceTargetDataset(Dataset):
    def __init__(self, train_dataset: Dataset, target_dataset: Dataset) -> None:
        self._train_dataset = train_dataset
        self._target_dataset = target_dataset

    def __len__(self):
        return len(self._train_dataset)

    def __getitem__(self, idx: int):
        train_batch = self._train_dataset[idx]
        target_batch = self._target_dataset[idx]
        return {
            "train_batch": train_batch,
            "target_batch": target_batch
        }
