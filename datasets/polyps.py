from os import listdir
from os.path import join

import albumentations as alb
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from torchvision import transforms as transforms
from torchvision.transforms import ToTensor

from utils import INPUT_SIZE


class KvasirSegmentationDataset(Dataset):
    """
        Dataset class that fetches images with the associated segmentation mask.
    """
    def __init__(self, path, train_alb, val_alb, split="train"):
        super(KvasirSegmentationDataset, self).__init__()
        self.path = path
        self.fnames = sorted(listdir(join(self.path,"segmented-images", "images")))

        self.split = split
        self.train_transforms = train_alb
        self.val_transforms = val_alb
        train_size = int(len(self.fnames) * 0.8)
        val_size = (len(self.fnames) - train_size) // 2
        test_size = len(self.fnames) - train_size - val_size
        self.fnames_train = self.fnames[:train_size]
        self.fnames_val = self.fnames[train_size:train_size + val_size]
        self.fnames_test = self.fnames[train_size + val_size:]
        self.split_fnames = None  # iterable for selected split
        if self.split == "train":
            self.size = train_size
            self.split_fnames = self.fnames_train
        elif self.split == "val":
            self.size = val_size
            self.split_fnames = self.fnames_val
        elif self.split == "test":
            self.size = test_size
            self.split_fnames = self.fnames_test
        else:
            raise ValueError("Choices are train/val/test")
        self.tensor = ToTensor()


    def __len__(self):
        # return 16 #debug
        return self.size

    def __getitem__(self, index):
        # img = Image.open(join(self.path, "segmented-images", "images/", self.split_fnames[index]))
        # mask = Image.open(join(self.path, "segmented-images", "masks/", self.split_fnames[index]))

        image = np.asarray(Image.open(join(self.path, "segmented-images", "images/", self.split_fnames[index])))
        mask =  np.asarray(Image.open(join(self.path, "segmented-images", "masks/", self.split_fnames[index])))
        if self.split=="train":
            image, mask = self.train_transforms(image=image, mask=mask).values()
        else:
            image, mask = self.val_transforms(image=image, mask=mask).values()
        image, mask = transforms.ToTensor()(Image.fromarray(image)), transforms.ToTensor()(Image.fromarray(mask))
        mask = torch.mean(mask,dim=0,keepdim=True).int()
        return image,mask, index


class EtisDataset(Dataset):
    """
        Dataset class that fetches Etis-LaribPolypDB images with the associated segmentation mask.
        Used for testing.
    """

    def __init__(self, path, trans):
        super(EtisDataset, self).__init__()
        self.path = path
        self.len = len(listdir(join(self.path, "Original")))
        self.transforms = trans
        self.tensor = ToTensor()

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        img_path = join(self.path, "Original/{}.jpg".format(i + 1))
        mask_path = join(self.path, "GroundTruth/p{}.jpg".format(i + 1))
        image = np.asarray(Image.open(img_path))
        mask = np.asarray(Image.open(mask_path))
        image, mask = self.transforms(image=image, mask=mask).values()

        return self.tensor(image), self.tensor(mask)[0].unsqueeze(0).int(), i


class CVC_ClinicDB(Dataset):
    def __init__(self, path, transforms):
        super(CVC_ClinicDB, self).__init__()

        self.path = path
        self.len = len(listdir(join(self.path, "Original")))
        indeces = range(self.len)
        self.train_indeces = indeces[:int(0.8*self.len)]
        self.val_indeces = indeces[int(0.8*self.len):]
        self.transforms = transforms
        self.common_transforms = transforms
        self.tensor = ToTensor()

    def __getitem__(self, i):
        img_path = join(self.path, "Original/{}.png".format(i+ 1))
        mask_path = join(self.path, "Ground Truth/{}.png".format(i + 1))
        image = np.asarray(Image.open(img_path))
        mask = np.asarray(Image.open(mask_path))
        image, mask = self.transforms(image=image, mask=mask).values()
        # mask = (mask>0.5).int()[0].unsqueeze(0)
        return self.tensor(image), self.tensor(mask)[0].unsqueeze(0).int(), i

    def __len__(self):
        # return 16 #debug
        #
        return self.len

class EndoCV2020(Dataset):
    def __init__(self, root_directory, trans):
        super(EndoCV2020, self).__init__()
        self.root = root_directory
        self.mask_fnames = sorted(listdir(join(self.root, "masksPerClass", "polyp")))
        self.mask_locs = [join(self.root, "masksPerClass", "polyp", i) for i in self.mask_fnames]
        self.img_locs = [join(self.root, "originalImages", i.replace("_polyp", "").replace(".tif", ".jpg")) for i in
                         self.mask_fnames]
        self.trans = trans
        self.tensor = ToTensor()


    def __getitem__(self, i):
        image = np.asarray(Image.open(self.img_locs[i]))
        mask = np.asarray(Image.open(self.mask_locs[i]))

        image, mask = self.trans(image=image, mask=mask).values()
        return self.tensor(image), self.tensor(mask)[0].unsqueeze(0).int(), i

    def __len__(self):
        # return 16 #debug
        return len(self.mask_fnames)


def build_polyp_dataset(root):
    train_trans = alb.Compose([alb.Resize(INPUT_SIZE, INPUT_SIZE), alb.HorizontalFlip(), alb.RandomRotate90(), alb.Transpose()])
    val_trans = alb.Compose([alb.Resize(INPUT_SIZE, INPUT_SIZE)])
    kvasir_root = join(root, "HyperKvasir")
    train_dataset = KvasirSegmentationDataset(kvasir_root, train_trans, val_trans, split="train")
    train_val = KvasirSegmentationDataset(kvasir_root, train_trans, val_trans, split="val")
    train_test = KvasirSegmentationDataset(kvasir_root, train_trans, val_trans, split="test")
    etis = EtisDataset(join(root, "ETIS-LaribPolypDB"), val_trans)
    cvc = CVC_ClinicDB(join(root, "CVC-ClinicDB"), val_trans)
    endo = EndoCV2020(join(root,"EndoCV2020"), val_trans)
    return train_dataset, train_val, train_test, etis, cvc, endo
