import os
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm

from components import OODDetector
import itertools

def load_all(batch_size=30, samples=100, shift="normal"):
    dfs = []
    for model in MODELS:
        for dataset in DATASETS:
            if dataset=="Polyp" and model not in SEG_MODELS:
                continue
            if dataset!="Polyp" and model in SEG_MODELS:
                continue
            for dsd in DSDS:
                dfs.append(load_data(dataset, dsd, batch_size=batch_size, samples=samples, model=model, shift=shift))
    return pd.concat(dfs)

def _feature_file_matches(fname, dataset_name, feature_name, shift):
    """Exact match on '<dataset>_<mode>_<detector>.csv'.

    Substring matching is unsafe: 'entropy' is contained in 'cross_entropy', and
    'Office' in both Office31 and OfficeHome. ``shift`` filters on the mode ('' = all).
    """
    if not fname.endswith(".csv") or not fname.startswith(f"{dataset_name}_"):
        return False
    rest = fname[len(dataset_name) + 1:-4]
    mode, _, feat = rest.partition("_")
    return feat == feature_name and shift in mode


def load_data(dataset_name, feature_name, batch_size=1, samples=1000, model="resnet", shift="normal"):
    prefix = f"data/{model}/feature_data"
    if dataset_name=="Polyp" and feature_name=="msp" or (dataset_name=="Polyp" and model not in SEG_MODELS) :
        return pd.DataFrame() #MSP does not work for segmentation
    try:
            df = pd.concat([pd.read_csv(join(prefix, fname)) for fname in sorted(os.listdir(prefix))
                            if _feature_file_matches(fname, dataset_name, feature_name, shift)])
    except:
        print(f"no data found for path {prefix}/{dataset_name}_{shift}_{feature_name}.csv")
        return pd.DataFrame()
    df["Dataset"]=dataset_name
    df["batch_size"]=batch_size
    df["Model"]=model
    # Older collections label the MSP detector "softmax" inside the CSV, newer ones (WILDS) "msp".
    # Normalize to "softmax", which the ATC / AC / DoC / Reg-* baselines key on.
    df["feature_name"] = df["feature_name"].replace({"msp": "softmax"})

    df["shift"] = df["fold"].apply(lambda x: x.split("_")[0] if "_0." in x else x)  # what kind of shift has occured?
    df["shift_intensity"] = df["fold"].apply(
        lambda x: x.split("_")[1] if "0." in x else "InD" if "ind" in x else "Train" if "train" in x else "OoD")  # what intensity?
    try:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    except:
        pass
    sampled_ind_val_loss = np.quantile(np.array([df[df["fold"] == "ind_val"]["loss"].sample(batch_size).mean() for _ in range(samples)]), 0.95)

    if dataset_name=="Polyp":
        df["correct_prediction"] = df["loss"] < 0.5  # arbitrary threshold
    else:
        df["correct_prediction"] = df["acc"]==1 #arbitrary threshold;

    df["ood"] = ~df["fold"].isin(["train", "ind_val", "ind_test"])

    df["shift"] = df["fold"].apply(
        lambda x: x.split("_")[0] if "_0." in x else x)  # what kind of shift has occured?
    df["shift_intensity"] = df["fold"].apply(
        lambda x: x.split("_")[
            1] if "0." in x else "InD" if "ind" in x else "Train" if "train" in x else "OoD")  # what intensity?
    df["batch_size"]=batch_size
    df["Organic"] = df["shift_intensity"].isin(["InD", "OoD"])
    return df


# "cross_entropy" is the loss against an all-ones target (-sum_c log p_c for classifiers), NOT
# predictive entropy; it is reported as "UniformCE". True predictive entropy is "entropy".
DSD_PRINT_LUT = {"grad_magnitude": "GradNorm", "cross_entropy": "UniformCE", "entropy": "Entropy",
                 "energy": "Energy", "knn": "kNN",
                 "msp": "MSP", "softmax": "MSP", "typicality": "Typicality",
                 "mahalanobis": "Mahalanobis", "react": "ReAct", "vim": "ViM"}
SEG_MODELS = ["deeplabv3plus", "unet", "segformer"]
MODELS = ["resnet", "vit"] + SEG_MODELS
# print name -> file key used by load_data ("softmax" is only the CSV column alias of "msp")
DSD_LUT = {value: key for key, value in DSD_PRINT_LUT.items() if key != "softmax"}
# The msp detector writes feature_name="softmax" into its CSVs while its files are named *_msp.csv.
FEATURE_COLUMN_ALIAS = {"msp": "softmax"}
DATASETS = ["CCT", "OfficeHome", "Office31", "NICO", "Camelyon17", "IWildCam", "Polyp"]
# The eight post-hoc detectors used in the paper. Typicality (Glow) was dropped: not post-hoc,
# and no flow models exist for the WILDS datasets. Its cached features remain under data/.
# "entropy" has no cached features until collect_entropy_and_softmax.py has been run on the GPU
# machine; loaders skip it gracefully until then.
DSDS = ["knn", "grad_magnitude", "cross_entropy", "entropy", "energy", "msp", "mahalanobis", "react", "vim"]
# Collected (ResNet only) but not part of the paper's benchmark set; ImageNet-R/-A accuracy
# additionally depends on 200-class logit masking that has not been verified.
IMAGENET_DATASETS = ["ImageNetV2", "ImageNetR", "ImageNetA"]
BATCH_SIZES = [1]
THRESHOLD_METHODS = ["val_optimal", "id_quantile"]

COLUMN_PRINT_LUT = {"feature_name":"Feature", "loss":"Loss", "rate":"p(E)", "shift_intensity":"Shift Intensity", "shift":"Shift", "feature": "Feature Value"}
SAMPLERS = ["RandomSampler",  "ClassOrderSampler", "ClusterSampler", "SequentialSampler",]
SYNTHETIC_SHIFTS = ["noise", "multnoise", "hue", "saltpepper", "brightness", "contrast", "smear", "fog", "jpeg"]

SHIFT_PRINT_LUT= {"normal": "Organic", "noise": "Additive Noise", "multnoise": "Multiplicative Noise",
             "hue": "Hue", "saltpepper": "Salt+Pepper Noise", "brightness":"Brightness", "contrast":"Contrast", "smear":"Smear", "fog":"Fog", "jpeg":"JPEG"}
INPUT_SIZE = 224
SHIFT_LUT = {value: key for key, value in SHIFT_PRINT_LUT.items()}
