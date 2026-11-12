import albumentations as alb
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils import data
import albumentations
import random

def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
def additive_noise(x, intensity):
    seed_all(0)
    noise = torch.randn_like(x) * intensity
    return x + noise

def contrast(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    desat = albumentations.ColorJitter(hue=0, brightness=0, saturation=0, contrast=intensity*3, always_apply=True)
    transforms = alb.Compose([desat])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed

def hue_shift(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    desat = albumentations.ColorJitter(hue=intensity*3, brightness=0, saturation=0, contrast=0, always_apply=True)
    transforms = alb.Compose([desat])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed

def brightness_shift(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    desat = albumentations.ColorJitter(brightness=intensity*3, hue=0, contrast=0, saturation=0, always_apply=True)
    transforms = alb.Compose([desat])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed

def snow(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    snow = albumentations.RandomSnow(snow_point_lower=int(intensity*100), snow_point_upper=int(intensity*200), always_apply=True)
    transforms = alb.Compose([snow])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed

def fog(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    fog = albumentations.RandomFog(fog_coef_lower=intensity, fog_coef_upper=intensity, always_apply=True)
    transforms = alb.Compose([fog])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed



def multiplicative_noise(x, intensity):
    seed_all(0)
    noise = 1+torch.randn_like(x) * intensity*2
    return x * noise

def salt_and_pepper(x, intensity, scale=0.25):
    seed_all(0)
    noise = torch.rand_like(x)
    x = x.clone()
    scaled=intensity*scale
    x[noise<scaled] = 0
    x[noise>1-scale] = 1
    return x

def smear(img, intensity, mask=None):
    """Grid distortion. A geometric warp: for segmentation, pass ``mask`` so the label is
    warped identically (returns (img, mask)); otherwise the mask no longer matches the image."""
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    desat = albumentations.GridDistortion(always_apply=True, distort_limit=intensity*2)
    transforms = alb.Compose([desat])
    if mask is None:
        return ToTensor()(transforms(image=x)["image"])
    m = mask.permute(1, 2, 0).numpy().astype(np.uint8)
    out = transforms(image=x, mask=m)
    m_out = torch.from_numpy(np.asarray(out["mask"]).reshape(m.shape)).permute(2, 0, 1).to(mask.dtype)
    return ToTensor()(out["image"]), m_out

def jpeg(x, intensity):
    seed_all(0)
    x = x.permute(1, 2, 0).numpy()
    transforms = alb.Compose([
        alb.ImageCompression(quality_lower = int(100-intensity*50),
                             quality_upper = int(100-intensity*50), always_apply=True)])

    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed



def random_occlusion(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    occlusion = albumentations.Cutout(int(intensity*100), max_h_size=int(x.shape[0]*0.1), max_w_size=int(x.shape[1]*0.1), always_apply=True)
    transforms = alb.Compose([occlusion])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed


class TransformedDataset(data.Dataset):
    #generic wrapper for adding noise to datasets
    def __init__(self, dataset, transform, transform_name, transform_param):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.transform_param = transform_param
        self.transform_name = transform_name

    def __getitem__(self, index):

        batch = self.dataset.__getitem__(index)
        x = batch[0]
        rest = batch[1:]
        is_mask = bool(rest) and torch.is_tensor(rest[0]) and rest[0].dim() == 3
        if self.transform is smear and is_mask:  # geometric shift: warp the segmentation mask too
            x, mask = self.transform(x, self.transform_param, mask=rest[0])
            rest = (mask, *rest[1:])
        else:
            x = self.transform(x, self.transform_param)
        x = torch.clip(x, 0, 1)
        # if index==0:
        #     plt.imshow(x.permute(1,2,0))
        #     plt.savefig(f"test_plots/{self.transform_name}_{self.transform_param}.png")
        #     plt.show()
        #     plt.close()
        return (x, *rest)

    def __str__(self):
        return f"{type(self.dataset).__name__}_{self.transform_name}_{str(self.transform_param)}"

    def __len__(self):
        # return 1000 #debug
        return self.dataset.__len__()


