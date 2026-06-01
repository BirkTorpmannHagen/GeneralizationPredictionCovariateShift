import sys
import os


import os
import json

import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
import sklearn


class CCT(Dataset):
    def __init__(self, path, train_transform, val_transform, fold="train"):
        self.data = []
        self.labels = []
        self.locations = []
        self.category_names = []
        self.category_labels = []
        self.data_dir = path
        self.extract_all_metadata()
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.fold = fold
        self.label_encoder = sklearn.preprocessing.LabelEncoder()

        ind_merged_paths = list(self.train_image_paths) + list(self.cis_val_image_paths)
        ind_merged_labels = list(self.train_labels) + list(self.cis_val_labels)

        #use custom splits
        ind_merged_paths = list(self.train_image_paths) + list(self.cis_val_image_paths)
        ind_merged_labels = list(self.train_labels) + list(self.cis_val_labels)

        #split the ind data
        train_paths, valtest_paths, train_labels, valtest_labels= train_test_split(ind_merged_paths, ind_merged_labels, test_size=0.2, random_state=42)
        val_paths, test_paths, val_labels,  test_labels = train_test_split(valtest_paths, valtest_labels, test_size=0.5, random_state=42)

        # ind_merged_paths = [ind_merged_paths[i] for i in permutation]a
        # ind_merged_labels = [ind_merged_labels[i] for i in permutation]
        # num_ind = len(ind_merged_paths)
        # ind_split = [0.8, 0.1, 0.1]

        ood_merged_paths = list(self.trans_val_image_paths) + list(self.trans_test_image_paths)
        ood_merged_labels = list(self.trans_val_labels) + list(self.trans_test_labels)
        ood_val_paths, ood_test_paths, ood_val_labels, ood_test_labels = train_test_split(ood_merged_paths, ood_merged_labels, test_size=0.5, random_state=42)



        if fold=="train":
            self.file_names = train_paths
            self.labels = train_labels
        elif fold=="val":
            self.file_names = val_paths
            self.labels = val_labels
        elif fold=="test":
            self.file_names = test_paths
            self.labels = test_labels
        elif fold=="ood_val":
            self.file_names = ood_val_paths
            self.labels = ood_val_labels
        elif fold=="ood_test":
            self.file_names = ood_test_paths
            self.labels = ood_test_labels
        self.num_classes = len(np.unique(self.labels))
        self.label_encoder.fit(np.unique(self.labels))  # classes seen during training

    def __len__(self):
        # return 64 #debug
        return len(self.file_names)


    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.fold=="train":
            img = self.train_transform(img)
        else:
            img = self.val_transform(img)
        label = self.label_encoder.transform([self.labels[index]])[0]
        return img, label, index

    def extract_all_metadata(self):
        def extract_metadata(path_to_json):
            with open(path_to_json) as json_file:
                data = json.load(json_file)

            # image_paths = np.array([os.path.join(self.data_dir,'small_images/cct_18_all_images_sm',str(item['image_id'])+'.jpg') for item in data['annotations']])
            image_paths = np.array(
                [os.path.join(self.data_dir, 'standard_images/', str(item['image_id']) + '.jpg') for
                 item in data['annotations']])
            labels = np.array([int(item['category_id']) for item in data['annotations']])
            self.cat_dict = {int(item['id']): str(item['name']) for item in data['categories']}
            image_paths = image_paths[labels != 30]
            labels = labels[labels != 30]  # not present in the training set
            locations = np.array([int(item['location']) for item in data['images']])
            category_names = np.array([str(item['name']) for item in data['categories']])
            category_labels = np.array([int(item['id']) for item in data['categories']])
            return image_paths, np.squeeze(labels), locations, category_labels, category_names


        self.cis_test_image_paths, self.cis_test_labels, self.cis_test_locations, self.cis_test_category_labels, self.cis_test_category_names = extract_metadata(
            os.path.join(self.data_dir, 'annotations/cis_test_annotations.json'))

        self.cis_val_image_paths, self.cis_val_labels, self.cis_val_locations, self.cis_val_category_labels, self.cis_val_category_names = extract_metadata(
            os.path.join(self.data_dir, 'annotations/cis_val_annotations.json'))

        self.trans_test_image_paths, self.trans_test_labels, self.trans_test_locations, self.trans_test_category_labels, self.trans_test_category_names = extract_metadata(
            os.path.join(self.data_dir, 'annotations/trans_test_annotations.json'))

        self.trans_val_image_paths, self.trans_val_labels, self.trans_val_locations, self.trans_val_category_labels, self.trans_val_category_names = extract_metadata(
            os.path.join(self.data_dir, 'annotations/trans_val_annotations.json'))

        self.train_image_paths, self.train_labels, self.train_locations, self.train_category_labels, self.train_category_names = extract_metadata(
            os.path.join(self.data_dir, 'annotations/train_annotations.json'))

def build_cct_dataset(path, train_transform, val_transform):
    train = CCT(path, train_transform, val_transform, fold="train")
    val = CCT(path, train_transform, val_transform, fold="val")
    test = CCT(path, train_transform, val_transform, fold="test")
    ood_val = CCT(path, train_transform, val_transform, fold="ood_val")
    ood_test = CCT(path, train_transform, val_transform, fold="ood_test")
    return train, val, test, ood_val, ood_test




if __name__ == '__main__':
    ind_train, ind_val, ind_test, ood_val, ood_test = build_cct_dataset("../../Datasets/CCT", )
