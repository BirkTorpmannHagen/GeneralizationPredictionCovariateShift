import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from pygam import LinearGAM
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import balanced_accuracy_score

from sklearn.neighbors import KernelDensity
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

counter=0
def l2_distance(a, b):
    """
    Compute the L2 distance between two vectors.
    :param a: First vector (numpy array)
    :param b: Second vector (numpy array)
    :return: L2 distance (float)
    """
    return np.linalg.norm(a - b)

def ks_distance(a, b):
    try:
        return ks_2samp(a[a!=0], b[b!=0]).pvalue
    except ValueError:
        return -1




def get_optimal_threshold(ind, ood):
    merged = np.concatenate([ind, ood])
    max_acc = 0
    threshold = 0
    if ind.mean()<ood.mean():
        higher_is_ood = True
    else:
        higher_is_ood = False

    #if linearly seperable, set the threshold to the middle
    if ind.max()<ood.min() and higher_is_ood:
        return (ind.max()+ood.min())/2
    if ood.max()<ind.min() and not higher_is_ood:
        return (ind.max() + ood.min()) / 2

    for t in np.linspace(merged.min(), merged.max(), 500):
        if higher_is_ood:
            ind_acc = (ind<t).mean()
            ood_acc = (ood>t).mean()
        else:
            ind_acc = (ind>t).mean()
            ood_acc = (ood<t).mean()
        bal_acc = 0.5*(ind_acc+ood_acc)
        if not higher_is_ood:
            if bal_acc>=max_acc: #ensures that the threshold is near ind data for highly seperable datasets
                max_acc = bal_acc
                threshold = t
        else:
            if bal_acc>max_acc:
                max_acc = bal_acc
                threshold = t


    return threshold


class OODDetector:

    def __init__(self, df, threshold_method="val_optimal", id_quantile=0.95):
        assert df["feature_name"].nunique() == 1, df["feature_name"].unique()
        assert df["Dataset"].nunique() == 1
        assert df["Model"].nunique() == 1
        if "k" in df.columns:
            assert df["k"].nunique() == 1, "OODDetector only works with k=1 due to the thresholding scheme"
        self.df = df.copy()
        self.threshold_method = threshold_method
        self.ind_val = self.df[~self.df["ood"]]
        self.ood_val = self.df[self.df["ood"]]
        assert not self.ind_val.empty, "No in-distribution validation data"
        assert not self.ood_val.empty, "No out-of-distribution validation data"
        self.counter = 0
        self.higher_is_ood = self.ood_val["feature"].mean() >self.ind_val["feature"].mean()

        if threshold_method == "val_optimal":
            self.threshold = get_optimal_threshold(self.ind_val["feature"], self.ood_val["feature"])
        if threshold_method == "id_quantile":
            # ID-only thresholding: threshold is the q-th quantile of InD scores; no OOD calibration.
            # Direction (higher_is_ood) is inferred from val data, but the threshold itself uses InD only.
            q = float(id_quantile)
            if self.higher_is_ood:
                self.threshold = float(self.ind_val["feature"].quantile(q))
            else:
                self.threshold = float(self.ind_val["feature"].quantile(1.0 - q))
        if threshold_method == "ind_span":
            lower = self.ind_val["feature"].quantile(0.01)
            upper = self.ind_val["feature"].quantile(0.99)
            self.threshold = [lower,upper]
        if threshold_method=="logistic":
            self.logreg = LogisticRegression()
            min_len = min(len(self.ind_val), len(self.ood_val))

            #avoid poor balancing
            if len(self.ind_val) > min_len:
                self.ind_val = self.ind_val.sample(min_len, random_state=42)
            if len(self.ood_val) > min_len:
                self.ood_val = self.ood_val.sample(min_len, random_state=42)

            features = np.concatenate([self.ind_val["feature"].values, self.ood_val["feature"].values]).reshape(-1, 1)
            labels = np.concatenate([self.ind_val["ood"].astype(int).values, self.ood_val["ood"].astype(int).values])
            self.logreg.fit(features, labels)





    def train_kde(self):
        sns.kdeplot(self.ind_val["feature"], label="ind_val", color="blue")
        sns.kdeplot(self.ood_val["feature"], label="ood_val", color="red")
        if self.threshold_method == "ind_span":
            plt.axvline(self.threshold[0], color="red", linestyle="--")
            plt.axvline(self.threshold[1], color="red", linestyle="--")
        elif self.threshold_method == "val_optimal":
            plt.axvline(self.threshold, color="red", linestyle="--")
        plt.show()

    def val_kde(self, dataset, fname="kde_plot.png"):
        assert dataset["ood"].nunique()==2, dataset["ood"]
        print(dataset.groupby("ood")["feature"].mean())
        sns.kdeplot(dataset, x=self.ind_val["feature"] ,hue="ood", common_norm=False, alpha=0.5)
        plt.axvline(self.threshold, color="red", linestyle="--")
        plt.savefig(fname)
        plt.show()

    def forward(self, batch):
        feature = batch["feature"]

        if self.threshold_method=="logistic":
            output = self.logreg.predict(np.array(feature))
            print(output)
            return output
        else:
            return self.predict(batch)

    def predict(self, batch):
        # if isinstance(batch["feature"], pd.Series): #if it is a batch
        #     feature = batch["feature"].mean()
        # else:
        #     feature = batch["feature"]
        feature = batch["feature"]
        if self.threshold_method == "ind_span":
            return not (self.threshold[0] <= feature <= self.threshold[1])
        elif self.threshold_method in ("val_optimal", "id_quantile"):
            if self.higher_is_ood:
                return  feature > self.threshold
            else:
                return feature < self.threshold
        elif self.threshold_method == "logistic":
            output = bool(self.logreg.predict(np.array(feature).reshape(1, -1))[0])
            return output
        else:
            raise NotImplementedError



    def get_tpr(self, data):
        subset = data[data["ood"]]
        if subset.empty:
            return 1
        tpr =  subset.apply(lambda row: self.predict(row), axis=1).mean()
        return tpr


    def get_tnr(self, data):
        subset = data[~data["ood"]]
        if subset.empty:
            return 1
        tnr =  1-subset.apply(lambda row: self.predict(row), axis=1).mean()
        return tnr

    def get_accuracy(self, data):
        return 0.5*(self.get_tpr(data)+self.get_tnr(data))
    def get_dr(self, data):

        return data.apply(lambda row: self.predict(row), axis=1).mean()


    def get_metrics(self, data):
        return self.get_tpr(data), self.get_tnr(data), self.get_accuracy(data)


    def plot_hist(self):
        self.counter+=1
        sns.histplot(self.df, x="feature", hue="ood", alpha=0.5)
        plt.title(f"{self.df['feature_name'].unique()[0]} - {self.df['Dataset'].unique()[0]}: {self.get_accuracy(self.df)}")

        if self.threshold_method == "ind_span":
            plt.axvline(self.threshold[0], color="red", linestyle="--")
            plt.axvline(self.threshold[1], color="red", linestyle="--")
        elif self.threshold_method == "val_optimal":
            plt.axvline(self.threshold, color="red", linestyle="--")

        # plt.title(self.get_accuracy(self.df))
        if os.path.exists("test_figs")==False:
            os.makedirs("test_figs")
        plt.savefig(f"test_figs/histogram_{self.counter}.png")
        plt.close()
        # plt.show()

    def get_likelihood(self):
        return self.get_tpr(self.df), self.get_tnr(self.df)


class DebiasedOODDetector(OODDetector):
    def __init__(self, df, ood_val_shift, threshold_method="val_optimal", k=5, batch_size=32, distance_metric=ks_distance):

        super().__init__(df, ood_val_shift, threshold_method)
        assert df["feature_name"].nunique() == 1
        assert df["Dataset"].nunique() == 1
        # assert df["batch_size"].unique() == 1, "DebiasedOODDetector only works with batch_size=1 due to k-nearest scheme"

        self.df = df
        self.threshold_method = threshold_method
        self.distance = distance_metric
        # self.ind_val = df[(df["shift"] == "ind_val")&(~df["ood"])]
        # self.ood_val = df[(df["shift"] == ood_val_shift)&(df["ood"])]
        self.ind_val = df[~df["ood"]]
        self.ood_val = df[df["ood"]]
        self.k=k

        #calibration using bootstrapping
        ind_bootstrap_dists = []
        ood_bootstrap_dists = []
        for _ in range(1000):
            ind_sample_batch = self.ind_val.sample(batch_size)["feature"] #randomly sample from ind_val
            k_nearest_for_each = np.array([self.ind_val["feature"].values[np.argpartition(np.abs(i-self.ind_val["feature"]), self.k)[:self.k]] for i in ind_sample_batch]).flatten() #get the k nearest neighbors in ind_val
            ind_bootstrap_dists.append(self.distance(ind_sample_batch.values, k_nearest_for_each))

            ood_sample_batch = self.ood_val.sample(batch_size)["feature"]  # randomly sample from ind_val
            k_nearest_for_each = np.array(
                [self.ind_val["feature"].values[np.argpartition(np.abs(i - self.ind_val["feature"]), self.k)[:self.k]]
                 for i in ood_sample_batch]).flatten()  # get the k nearest neighbors in ind_val
            ood_bootstrap_dists.append(self.distance(ood_sample_batch.values, k_nearest_for_each))


        self.threshold = np.min(ind_bootstrap_dists) #set the threshold to the 90th percentile of the bootstrap distribution
        # self.threshold
        # self.threshold
        # self.threshold = (np.min(ind_bootstrap_dists) + np.max(ood_bootstrap_dists))/2 # middle between the min of ind and max of ood bootstrap distances
        # print(self.threshold)

        # self.kde = KernelDensity(kernel='gaussian')
        # self.kde.fit(np.array(bootstrap_dists).reshape(-1, 1))
        # self.batch_size = batch_size
        # ps =  np.exp(self.kde.score_samples(np.array(bootstrap_dists).reshape(-1, 1))) #set the threshold to the mean of the bootstrap distribution
        # print(ps)
        # input()

        # sns.kdeplot(bootstrap_dists, label="bootstrap distances", color="blue")
        # plt.show()

        # self.threshold = np.quantile(bootstrap_dists, 0.9) #set the threshold to the 95th percentile of the bootstrap distribution

    def predict(self, batch):
        assert batch["ood"].nunique() == 1, "Batch should have a single ood value"
        dists = []
        features = batch["feature"].values
        k_nearest_for_each = np.array(
            [self.ind_val["feature"].values[np.argpartition(np.abs(i - self.ind_val["feature"]), self.k)[:self.k]] for i
             in features]).flatten()  # get the k nearest neighbors in ind_val
        ks = self.distance(features, k_nearest_for_each)
        verdict = ks<self.threshold #if the distance is greater than the threshold, then it is ood
        # print(ks)
        # print(batch)

        # print(f"{batch['fold'].unique()[0]}: {ks}<{self.threshold}: Correct={(verdict == batch['ood'].all())}")
        # input()

        return verdict



class SyntheticOODDetector:
    def __init__(self, tpr, tnr):
        self.tpr = tpr
        self.tnr = tnr


    def predict(self, batch):

        # print(batch.columns)
        # input()
        if type(batch["ood"])==bool:
            if batch["ood"]:
                return 1 if np.random.rand() < self.tpr else 0
            else:
                return 0 if np.random.rand() < self.tnr else 1

        assert batch["ood"].nunique()==1
        if batch["ood"].all(): #if the sample is ood
            #return true with likelihood = tpr, else false (1-tpr)
            return 1 if np.random.rand() < self.tpr else 0
        else: #if the sample is ind
            #return true with likelihood = tnr, else false (1-tnr)
            return 0 if np.random.rand() < self.tnr else 1

    def get_likelihood(self):
        return self.tpr, self.tnr


class EnsembleOODDetector(OODDetector):
    def __init__(self, df, detector_list):
        self.detector_list = detector_list
        self.meta_head = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000)
        )
        X = df[detector_list].values
        y = df["ood"].astype(int).values
        self.meta_head.fit(X, y)

        # choose threshold on this training/val split to maximize BA
        proba = self.meta_head.predict_proba(X)[:, 1]
        fpr, tpr, thr = roc_curve(y, proba)
        ba = (tpr + (1 - fpr)) / 2
        print(max(ba))
        self.threshold_ = float(thr[np.argmax(ba)])

    def predict(self, batch):
        # batch is a Series (row) or dict with the feature names as keys
        X = pd.DataFrame([batch])[self.detector_list]  # <-- align by index, THEN pick columns
        proba = self.meta_head.predict_proba(X)[:, 1]
        return bool(proba[0] >= self.threshold_)

    def get_balanced_accuracy(self, df):
        """
        Compute balanced accuracy on a given dataframe.
        """
        X = df[self.detector_list].values
        y_true = df["ood"].astype(int).values
        y_pred = (self.meta_head.predict_proba(X)[:, 1] >= self.threshold_).astype(int)
        return balanced_accuracy_score(y_true, y_pred)

    def show_weights(self, original_scale=True, sort=True):
        """
        Display the logistic regression coefficients from the meta-head.

        Parameters
        ----------
        original_scale : bool, optional
            If True, rescales coefficients back to the original (unstandardized) feature scale.
            If False, shows the weights learned on standardized features.
        sort : bool, optional
            If True, sort by absolute weight magnitude.
        """
        logreg = self.meta_head.named_steps["logisticregression"]
        scaler = self.meta_head.named_steps["standardscaler"]

        if original_scale:
            coefs = logreg.coef_.ravel() / scaler.scale_
            intercept = logreg.intercept_ - np.sum(
                (scaler.mean_ / scaler.scale_) * logreg.coef_.ravel()
            )
        else:
            coefs = logreg.coef_.ravel()
            intercept = logreg.intercept_

        weights = pd.Series(coefs, index=self.detector_list)
        if sort:
            weights = weights.reindex(weights.abs().sort_values(ascending=False).index)

        print("Intercept:", intercept[0] if hasattr(intercept, "__len__") else intercept)
        print("\nFeature Weights:")
        print(weights.to_string(float_format=lambda x: f"{x: .4f}"))

        return weights, intercept


class LogisticRiskCalibrator:
    """
    Learn g(s) = P(error | OOD score s) with logistic regression,
    then abstain when g(s) >= epsilon.

    Expects a calibration dataframe with:
      - 'feature': OOD score (scalar)
      - 'correct_prediction': 1 if model was correct, 0 if incorrect
    """

    def __init__(self, C=1.0, max_iter=2000):
        self.lr = LogisticRegression(class_weight="balanced", C=C, max_iter=max_iter)
        self.fitted_ = False

    def fit(self, calib_df: pd.DataFrame):
        """Fit logistic regression on calibration pairs (score -> is_error)."""
        assert "feature" in calib_df.columns and "correct_prediction" in calib_df.columns
        X = calib_df["feature"].values.reshape(-1, 1)
        # Convert to is_error = 1 - correct_prediction
        y = (1 - calib_df["correct_prediction"].astype(int)).values
        self.lr.fit(X, y)
        self.fitted_ = True
        return self

    def predict_error_prob(self, score):
        """Map OOD score -> P(error). Works on scalar or array-like."""
        assert self.fitted_, "Call fit() first."
        s = np.asarray(score).reshape(-1, 1)
        p = self.lr.predict_proba(s)[:, 1]
        return p if p.size > 1 else float(p[0])

    def abstain(self, score, epsilon=0.10):
        """
        Return True (abstain) if predicted error >= epsilon; False otherwise.
        Accepts scalar score; for arrays, returns a boolean array.
        """
        p = self.predict_error_prob(score)
        return (p >= float(epsilon)) if isinstance(p, np.ndarray) else bool(p >= float(epsilon))

    # --- Evaluation helpers (on a labeled test set) ---

    def risk_coverage(self, test_df: pd.DataFrame, epsilons=None):
        """
        Compute realized risk vs coverage on a labeled test set.
        test_df columns: 'feature', 'correct_prediction' (1/0)
        Returns a DataFrame with columns: epsilon, coverage, realized_risk.
        """
        assert self.fitted_, "Call fit() first."
        if epsilons is None:
            epsilons = np.linspace(0.01, 0.5, 25)

        S = test_df["feature"].values
        # 1 - correct_prediction = 1 if error
        E = (1 - test_df["correct_prediction"].astype(int)).values
        P = self.predict_error_prob(S)  # vector

        rows = []
        for eps in epsilons:
            keep = P < eps
            coverage = float(keep.mean())
            realized_risk = float(E[keep].mean()) if keep.any() else 0.0
            rows.append(dict(epsilon=float(eps),
                             coverage=coverage,
                             realized_risk=realized_risk))
        return pd.DataFrame(rows)

    def choose_threshold_for_target_risk(self, calib_df: pd.DataFrame, target_epsilon=0.10):
        """
        On the calibration set, return the p_err threshold tau s.t.
        realized risk among kept samples ≈ target_epsilon.
        Use this tau at runtime: abstain if g(score) >= tau.
        """
        S = calib_df["feature"].values
        E = (1 - calib_df["correct_prediction"].astype(int)).values
        P = self.predict_error_prob(S)

        candidates = np.unique(np.concatenate([[0.0], P, [1.0]]))
        best_tau, best_gap = 0.0, float("inf")
        for tau in candidates:
            keep = P < tau
            if not keep.any():
                continue
            risk = E[keep].mean()
            gap = abs(risk - target_epsilon)
            if gap < best_gap:
                best_tau, best_gap = float(tau), gap
        return best_tau

class Trace:
    """
    Container for dsd verdict traces; used to estimate lambda
    """
    def __init__(self, trace_length=100):
        self.trace_length = trace_length
        self.trace = []

    def update(self, item):
        if len(self.trace) == self.trace_length:
            self.trace.pop(0)
        self.trace.append(item)
        return self.trace

