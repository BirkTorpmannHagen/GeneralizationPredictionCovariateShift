import pandas as pd

from tqdm import tqdm
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from multiprocessing import Pool
from components import ks_distance

# kNN / Mahalanobis / ViM reference database cap. The InD-train set is used only
# as a *reference* for distance/density detectors (not as an evaluation fold);
# subsampling it is standard (Sun et al. 2022) and keeps memory + the per-mode
# reference encoding tractable. Evaluation folds are never subsampled. Only
# datasets with more than this many training images are affected.
REFERENCE_CAP = 50000


def list_to_str(some_list):
    return "".join([i.__name__ for i in some_list])



def convert_to_pandas_df(train_features, train_losses,
                         ind_val_features, ind_val_losses,
                         ind_test_features, ind_test_losses,
                         ood_features, ood_losses,
                         feature_names):
    def add_entries(dataset, features_dict, losses_dict, fold_label_override=None):
        for fold, features in features_dict.items():
            losses = losses_dict[fold]
            label = fold_label_override if fold_label_override else fold
            data_len = max(len(features), len(losses)) #they removed a few data from the CCT dataset, this fixes it
            for i in range(data_len):
                try:
                    dataset.append({
                        "fold": label,
                        "feature_name": feature_name,
                        "feature": features[i][fi],
                        "loss": losses[i][0],
                        "acc": losses[i][1],
                        "idx": losses[i][2],
                        "class": losses[i][3] if losses.shape[1]>3 else None,
                    })
                except IndexError:
                    try:
                        dataset.append({
                            "fold": label,
                            "feature_name": feature_name,
                            "feature": features[i],
                            "loss": losses[i][0],
                            "acc": losses[i][1],
                            "idx": losses[i][2],
                            "class": losses[i][3] if losses.shape[1] > 3 else None,
                        })
                    except IndexError:
                        break

    dataframes = []

    for fi, feature_name in enumerate(feature_names):
        dataset = []
        add_entries(dataset, train_features, train_losses, fold_label_override="train")
        add_entries(dataset, ind_val_features, ind_val_losses)
        add_entries(dataset, ind_test_features, ind_test_losses)
        add_entries(dataset, ood_features, ood_losses)
        dataframes.append(pd.DataFrame(dataset))

    return dataframes

def convert_to_pandas_df_no_ind(
                         ood_features, ood_losses,
                         feature_names):
    def add_entries(dataset, features_dict, losses_dict, fold_label_override=None):
        for fold, features in features_dict.items():
            losses = losses_dict[fold]
            assert len(losses)==len(features)
            label = fold_label_override if fold_label_override else fold
            for i in range(features.shape[0]):
                try:
                    dataset.append({
                        "fold": label,
                        "feature_name": feature_name,
                        "feature": features[i][fi],
                        "loss": losses[i][0],
                        "acc": losses[i][1],
                        "idx": losses[i][2],
                        "class": losses[i][3] if losses.shape[1]>3 else None,
                    })
                except IndexError:
                    dataset.append({
                        "fold": label,
                        "feature_name": feature_name,
                        "feature": features[i],
                        "loss": losses[i][0],
                        "acc": losses[i][1],
                        "idx": losses[i][2],
                        "class": losses[i][3] if losses.shape[1] > 3 else None,
                    })

    dataframes = []

    for fi, feature_name in enumerate(feature_names):
        dataset = []
        add_entries(dataset, ood_features, ood_losses)
        dataframes.append(pd.DataFrame(dataset))

    return dataframes

class BaseSD:
    def __init__(self, rep_model):
        self.rep_model = rep_model

    def register_testbed(self, testbed):
        self.testbed = testbed


class FeatureSD(BaseSD):
    def __init__(self, rep_model, feature_fns):
        super().__init__(rep_model)
        self.feature_fns = feature_fns
        self.num_features=len(feature_fns)



    def save(self, var, varname):
        pkl.dump(var, open(f"cache_{list_to_str(self.feature_fns)}_{self.testbed.__class__.__name__}_{varname}.pkl", "wb"))

    def load(self, varname):
        return pkl.load(open(f"cache_{list_to_str(self.feature_fns)}_{self.testbed.__class__.__name__}_{varname}.pkl", "rb"))

    def get_features(self, dataloader):
        features = np.zeros((len(dataloader), self.testbed.batch_size, self.num_features))
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing Features"):
            x = data[0].cuda()
            for j, feature_fn in enumerate(self.feature_fns):
                # print(feature_fn)
                with torch.no_grad():
                    if feature_fn.__name__=="typicality":
                        features[i,:, j]=feature_fn(self.testbed.glow, x, self.train_test_encodings).detach().cpu().numpy()
                    else:
                        features[i,:, j]=feature_fn(self.testbed.classifier, x, self.train_test_encodings).cpu().numpy()
        features = features.reshape((len(dataloader)*self.testbed.batch_size, self.num_features))
        return features

    def _capped_reference_loader(self):
        """InD-train loader capped at REFERENCE_CAP for detector reference/fitting.

        Evaluation folds are untouched; only the (non-evaluation) reference set is
        subsampled, and only when the training set exceeds the cap."""
        ds = self.testbed.ind_train
        n = len(ds)
        if n > REFERENCE_CAP:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, REFERENCE_CAP, replace=False)
            ds = Subset(ds, idx.tolist())
            print(f"[FeatureSD] capping kNN/Maha/ViM reference {n} -> {REFERENCE_CAP}")
        return DataLoader(ds, batch_size=self.testbed.batch_size, shuffle=True,
                          num_workers=self.testbed.num_workers, drop_last=True)

    def get_encodings(self, dataloader):

        features = np.zeros((len(dataloader), self.testbed.batch_size, self.testbed.classifier.latent_dim))
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing Encodings"):
            x = data[0].cuda()
            with torch.no_grad():
                out = self.testbed.classifier.get_encoding(x).detach().cpu().numpy()
            features[i]=out
        features = features.reshape((len(dataloader)*self.testbed.batch_size, self.rep_model.latent_dim))
        # print(features)
        return features



    def compute_features_and_loss_for_loaders(self, dataloaders):

        losses = dict(
            zip(dataloaders.keys(),
                [self.testbed.compute_losses(loader) for fold_name, loader in dataloaders.items()]))
        features = dict(
            zip(dataloaders.keys(),
                          [self.get_features(loader)
                           for fold_name, loader in dataloaders.items()]))


        return features, losses

    def compute_pvals_and_loss(self, noind=False):
        """

        :param sample_size: sample size for the tests
        :return: ind_p_values: p-values for ind fold for each sampler
        :return ood_p_values: p-values for ood fold for each sampler
        :return ind_sample_losses: losses for each sampler on ind fold, in correct order
        :return ood_sample_losses: losses for each sampler on ood fold, in correct order
        """

        #these features are necessary to compute before-hand in order to compute knn and typicality
        # Reference (InD-train) is capped for the distance/density detectors and for
        # the (non-evaluation) train fold; all evaluation folds remain full.
        ref_loader = self._capped_reference_loader()
        if not noind:
            self.train_test_encodings = self.get_encodings(ref_loader)
            train_features, train_loss = self.compute_features_and_loss_for_loaders({"ind_train": ref_loader})
            ind_val_features, ind_val_losses = self.compute_features_and_loss_for_loaders(self.testbed.ind_val_loader())
            ind_test_features, ind_test_losses = self.compute_features_and_loss_for_loaders(self.testbed.ind_test_loader())
            ood_features, ood_losses = self.compute_features_and_loss_for_loaders(self.testbed.ood_loaders())
            return train_features, train_loss, ind_val_features,ind_val_losses, ind_test_features, ind_test_losses, ood_features,  ood_losses
        else:
            self.train_test_encodings = self.get_encodings(ref_loader)
            ood_features, ood_losses = self.compute_features_and_loss_for_loaders(self.testbed.ood_loaders())
            return ood_features,  ood_losses

class BatchedFeatureSD(FeatureSD):
    def __init__(self, rep_model, feature_fns):
        super().__init__(rep_model, feature_fns)
        self.feature_names = [fn.__name__ for fn in feature_fns]

    def compute_pvals_and_loss(self):
        indloaders = self.testbed.ind_loader()
        try:
            self.train_test_encodings = np.load(f"cache_{self.testbed.__class__.__name__}_train_test_encodings.npy")
            self.train_features_raw = np.load(f"cache_{self.feature_names}_{self.testbed.__class__.__name__}_train_features.npy")
            print("successfully loaded ind")
        except FileNotFoundError:
            print("Collecting ind data")
            self.train_test_encodings = super(BatchedFeatureSD, self).get_encodings(indloaders["ind_train"]).reshape((len(self.testbed.ind_loader()["ind_train"])*self.testbed.batch_size, self.rep_model.latent_dim))
            self.train_features_raw = super(BatchedFeatureSD, self).get_features(indloaders["ind_train"])
            np.save(f"cache_{self.testbed.__class__.__name__}_train_test_encodings.npy", self.train_test_encodings)
            np.save(f"cache_{self.feature_names}_{self.testbed.__class__.__name__}_train_features.npy", self.train_features_raw)
        self.train_features = {"ind_train": self.train_features_raw}
        train_loss = dict(
            zip(indloaders.keys(),
                [self.testbed.compute_losses(loader) for fold_name, loader in indloaders.items()]))
        ind_val_features, ind_val_losses = self.compute_features_and_loss_for_loaders(self.testbed.ind_val_loader())
        ind_test_features, ind_test_losses = self.compute_features_and_loss_for_loaders(self.testbed.ind_test_loader())
        ood_features, ood_losses = self.compute_features_and_loss_for_loaders(self.testbed.ood_loaders())
        return self.train_features, train_loss, ind_val_features,ind_val_losses, ind_test_features, ind_test_losses, ood_features,  ood_losses


    def get_features(self, dataloader):
        features = np.zeros((len(dataloader), self.num_features))
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Computing Features"):
                x = data[0].cuda()
                features_batch = np.zeros((self.num_features, self.testbed.batch_size))
                for j, feature_fn in enumerate(self.feature_fns):
                    if feature_fn.__name__ == "typicality":
                        features_batch[j] = feature_fn(self.testbed.glow, x,
                                                       self.train_test_encodings).detach().cpu().numpy()
                    else:
                        features_batch[j] = feature_fn(self.rep_model, x,
                                                       self.train_test_encodings).detach().cpu().numpy()

                with Pool(self.num_features) as pool:
                    results = pool.starmap(ks_distance, zip([features_batch[k] for k in range(self.num_features)],
                                                        [self.train_features["ind_train"][k] for k in
                                                         range(self.num_features)]))
                features[i]=results
        return features

    def compute_features_and_loss_for_loaders(self, dataloaders):

        losses = dict(
            zip(dataloaders.keys(),
                [self.testbed.compute_losses(loader, reduce=True) for fold_name, loader in dataloaders.items()]))
        features = dict(
            zip(dataloaders.keys(),
                          [self.get_features(loader)
                           for fold_name, loader in dataloaders.items()]))


        return features, losses

    def get_encodings(self, dataloader):

        features = np.zeros((len(dataloader), self.testbed.batch_size, self.rep_model.latent_dim))
        if self.testbed.batch_size==1:
            features = np.zeros((len(dataloader), self.rep_model.latent_dim))
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            x = data[0].cuda()
            with torch.no_grad():
                features[i] = self.rep_model.get_encoding(x).detach().cpu().numpy()
        return features

