import os

from numpy.lib.index_tricks import index_exp
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision.datasets import ImageFolder


class Office31Dataset(Dataset):
    def __init__(self, path, train_transform, val_transform, ind_context="amazon", fold="train"):
        super().__init__()
        self.path = path
        self.ind_context=ind_context

        self.ood_contexts = os.listdir(path)
        self.ood_contexts.remove(ind_context)
        self.num_classes = len(os.listdir(os.path.join(path, self.ind_context, "images")))

        # Load the full dataset without splits
        # Prepare labels for stratified split
        full_ind_dataset = ImageFolder(os.path.join(path, self.ind_context, "images"))
        ind_targets = [s[1] for s in full_ind_dataset.samples]  # Extract labels from samples
        # Perform stratified split
        ind_train_idx, ind_val_idx = train_test_split(
            range(len(full_ind_dataset)),
            test_size=0.2,
            random_state=42  # Ensures determinism
        )
        ind_test_idx = ind_val_idx[:len(ind_val_idx)//2]
        ind_val_idx = ind_val_idx[len(ind_val_idx)//2:]


        if fold == "train":
            self.dataset = Subset(full_ind_dataset, ind_train_idx)
            self.dataset.dataset.transform = train_transform
        elif fold=="val":
            self.dataset = Subset(full_ind_dataset, ind_val_idx)
            self.dataset.dataset.transform = val_transform
        elif fold=="test":
            self.dataset = Subset(full_ind_dataset, ind_test_idx)
            self.dataset.dataset.transform = val_transform
        elif fold=="ood_val":
            self.dataset = ImageFolder(os.path.join(path, self.ood_contexts[0], "images"))
            self.dataset.transform = val_transform
        elif fold=="ood_test":
            self.dataset = ImageFolder(os.path.join(path, self.ood_contexts[1], "images"))
            self.dataset.transform = val_transform

    def __len__(self):
        # return 64 #debug
        return len(self.dataset)

    def __getitem__(self, index):
        return *self.dataset[index], index


def build_office31_dataset(root, train_transform, val_transform, ind_context="amazon"):
    train = Office31Dataset(root, train_transform, val_transform, ind_context, fold="train")
    val = Office31Dataset(root, train_transform, val_transform, ind_context, fold="val")
    test = Office31Dataset(root, train_transform, val_transform, ind_context, fold="test")
    ood_val = Office31Dataset(root, train_transform, val_transform, ind_context, fold="ood_val")
    ood_test = Office31Dataset(root, train_transform, val_transform, ind_context, fold="ood_test")
    return train, val, test, ood_val, ood_test

