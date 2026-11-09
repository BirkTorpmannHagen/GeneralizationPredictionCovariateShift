import json
import os
from glob import glob
import random

from PIL import Image
from torch.utils import data


class NICODataset(data.Dataset):
    def __init__(self, image_path_list, label_map_json, transform):
        super().__init__()
        self.image_path_list = image_path_list
        self.transform = transform
        self.num_classes = len(os.listdir(os.path.join(*image_path_list[0].split("/")[:-2])))
        self.classes = sorted(os.listdir(os.path.join(*image_path_list[0].split("/")[:-2])))
        with open(label_map_json, "r") as f:
            self.label_map = json.load(f)
        context_path = os.path.join(*image_path_list[0].split("/")[:-3])
        self.context = image_path_list[0].split("/")[-3]
        contexts = sorted(os.listdir(context_path))
        self.context_map = dict(zip(contexts, range(len(contexts))))

    def __len__(self):
        # return 32 #debug
        return len(self.image_path_list)
    #
    def __getitem__(self, index):
        image_path = self.image_path_list[index]
        image = Image.open(image_path)
        image = self.transform(image)
        label = self._get_label_index(image_path)
        return image, label, index

    def _get_label_index(self, image_path):
        class_name = image_path.split("/")[-2]
        label_index = self.label_map[class_name]
        return label_index


    def fetch_model(self):
        """
        :return: trained classifier, pytorch lightning
        """
        pass


def build_nico_dataset(root, train_transform, val_transform, ind_context, seed=42):
    track_data_dir = os.path.join(root, "track_1")
    data_dir = os.path.join(track_data_dir, "public_dg_0416", "train")
    label_map_json = os.path.join(track_data_dir, "dg_label_id_mapping.json")
    # Sorted + seeded: the InD train/val/test split must be identical between classifier
    # training and every later feature collection. The previous unseeded shuffle re-drew the
    # split on every build, so "ind_val"/"ind_test" largely overlapped the classifier's
    # training images. Changing this requires retraining the NICO classifiers.
    rng = random.Random(seed)
    ind_image_path_list = sorted(glob(f"{data_dir}/{ind_context}/*/*.jpg"))

    context_path = os.path.join(*ind_image_path_list[0].split("/")[:-3])
    ood_contexts = sorted(os.listdir(context_path))
    ood_contexts.remove(ind_context)

    # One list per context, kept aligned with ood_contexts. (Previously the list of lists was
    # shuffled after the names were fixed, so domain names were assigned to the wrong images.)
    ood_image_path_list = [sorted(glob(f"{data_dir}/{context}/*/*.jpg")) for context in ood_contexts]

    rng.shuffle(ind_image_path_list)
    train_idx = int(len(ind_image_path_list) * 0.8)
    val_idx = int(len(ind_image_path_list) * 0.9)
    train = NICODataset(ind_image_path_list[:train_idx], label_map_json, train_transform)
    val = NICODataset(ind_image_path_list[train_idx:val_idx], label_map_json, val_transform)
    test = NICODataset(ind_image_path_list[val_idx:], label_map_json, val_transform)
    ood_dict = dict(zip(ood_contexts, [NICODataset(paths, label_map_json, val_transform) for paths in ood_image_path_list]))
    print(ood_dict)
    return train, val, test, ood_dict
