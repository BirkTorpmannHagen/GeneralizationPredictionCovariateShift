import os
from os.path import join

import numpy as np
import pandas as pd
from tqdm import tqdm

from components import OODDetector
import itertools

def load_all(batch_size=30, samples=100, shift="normal", pretrain=True):
    dfs = []
    for model in MODELS:
        for dataset in DATASETS:
            if dataset=="Polyp" and model not in SEG_MODELS:
                continue
            if dataset!="Polyp" and model in SEG_MODELS:
                continue
            for dsd in DSDS:
                dfs.append(load_data(dataset, dsd, batch_size=batch_size, samples=samples, model=model, shift=shift, pretrain=pretrain))
    return pd.concat(dfs)

def load_data(dataset_name, feature_name, batch_size=1, samples=1000, model="resnet", shift="normal"):
    prefix = f"data/pretrain/{model}/feature_data"
    if dataset_name=="Polyp" and feature_name=="msp" or (dataset_name=="Polyp" and model not in SEG_MODELS) :
        return pd.DataFrame() #MSP does not work for segmentation
    try:
            df = pd.concat([pd.read_csv(join(prefix, fname)) for fname in os.listdir(prefix) if dataset_name in fname and feature_name in fname and shift in fname])
    except:
        print(f"no data found for path {prefix}/{dataset_name}_{shift}_{feature_name}.csv")
        return pd.DataFrame()
    df["Dataset"]=dataset_name
    df["batch_size"]=batch_size
    df["Model"]=model

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


DSD_PRINT_LUT = {"grad_magnitude": "GradNorm", "cross_entropy": "Entropy", "energy": "Energy", "knn": "kNN", "msp": "MSP", "typicality": "Typicality"}
SEG_MODELS = ["deeplabv3plus", "unet", "segformer"]
MODELS = ["resnet", "vit"] + SEG_MODELS
DSD_LUT = {value: key for key, value in DSD_PRINT_LUT.items()}
DATASETS = ["CCT", "OfficeHome", "Office31", "NICO", "Polyp"]
DSDS = ["knn", "grad_magnitude", "cross_entropy", "energy", "typicality", "msp"]
BATCH_SIZES = [1]
THRESHOLD_METHODS = ["val_optimal", "id_quantile"]

COLUMN_PRINT_LUT = {"feature_name":"Feature", "loss":"Loss", "rate":"p(E)", "shift_intensity":"Shift Intensity", "shift":"Shift", "feature": "Feature Value"}
SAMPLERS = ["RandomSampler",  "ClassOrderSampler", "ClusterSampler", "SequentialSampler",]
SYNTHETIC_SHIFTS = ["noise", "multnoise", "hue", "saltpepper", "brightness", "contrast", "smear", "fog", "jpeg"]

SHIFT_PRINT_LUT= {"normal": "Organic", "noise": "Additive Noise", "multnoise": "Multiplicative Noise",
             "hue": "Hue", "saltpepper": "Salt+Pepper Noise", "brightness":"Brightness", "contrast":"Contrast", "smear":"Smear", "fog":"Fog", "jpeg":"JPEG"}
INPUT_SIZE = 224
SHIFT_LUT = {value: key for key, value in SHIFT_PRINT_LUT.items()}
