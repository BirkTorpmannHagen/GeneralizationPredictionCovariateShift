import os

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision.datasets import ImageFolder


class OfficeHome(Dataset):
    def __init__(self, path, train_transform, val_transform, context="Real World", fold="train"):
        super().__init__()
        self.path = path
        self.contexts = os.listdir(path)
        self.num_classes = len(os.listdir(os.path.join(path, self.contexts[0])))

        # Load the full dataset without splits
        full_dataset = ImageFolder(os.path.join(path, context))

        # Prepare labels for stratified split
        targets = [s[1] for s in full_dataset.samples]  # Extract labels from samples

        # Perform stratified split
        train_idx, val_idx = train_test_split(
            range(len(full_dataset)),
            test_size=0.2,
            stratify=targets,
            random_state=42  # Ensures determinism
        )
        test_idx = val_idx[:len(val_idx) // 2]
        val_idx = val_idx[len(val_idx) // 2:]

        if fold == "train":
            self.dataset = Subset(full_dataset, train_idx)
            self.dataset.dataset.transform = train_transform
        elif fold == "val":
            self.dataset = Subset(full_dataset, val_idx)
            self.dataset.dataset.transform = val_transform
        elif fold == "test":
            self.dataset = Subset(full_dataset, test_idx)
            self.dataset.dataset.transform = val_transform
        elif fold == "ood":
            self.dataset = full_dataset
            self.dataset.transform = val_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return *self.dataset[index], index


def build_officehome_dataset(root, train_transform, val_transform, context="Real World"):
    train = OfficeHome(root, train_transform, val_transform, context, fold="train")
    val = OfficeHome(root, train_transform, val_transform, context, fold="val")
    test = OfficeHome(root, train_transform, val_transform, context, fold="test")

    ood_contexts = train.contexts
    ood_contexts.remove(context)
    oods = dict(zip(ood_contexts, [OfficeHome(root, train_transform, val_transform, context, fold="val") for context in ood_contexts]))
    return train, val,test, oods
