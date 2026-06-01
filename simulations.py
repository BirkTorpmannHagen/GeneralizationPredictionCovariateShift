import numpy as np
import pandas as pd
from decimal import getcontext

from components import Trace, OODDetector
from rateestimators import ErrorAdjustmentEstimator, SimpleEstimator
from riskmodel import DetectorEventTree, BaseEventTree

np.set_printoptions(precision=4, suppress=True)
getcontext().prec = 4
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option("display.max_rows", None)

# datasets
from utils import *


class Simulator:
    """
    Abstract simulator class
"""

    def __init__(self, df, ood_test_fold, ood_val_fold, estimator=ErrorAdjustmentEstimator, trace_length=100,
                 **kwargs):

        self.df = df

        self.ood_test_fold= ood_test_fold
        if isinstance(ood_val_fold, str):
            self.ood_val_folds = [ood_val_fold]
        else:
            self.ood_val_folds = list(ood_val_fold)
        # back-compat alias for existing call sites that read .ood_val_fold
        self.ood_val_fold = self.ood_val_folds[0] if len(self.ood_val_folds) == 1 else self.ood_val_folds

        train_df = df[df["fold"].isin(self.ood_val_folds) | (df["fold"] == "ind_val")]

        self.ood_detector = OODDetector(train_df, "val_optimal")
        dsd_tpr, dsd_tnr = self.ood_detector.get_likelihood()


        self.ood_val_acc = self.get_predictor_accuracy(self.ood_val_folds)
        self.ood_test_acc = self.get_predictor_accuracy(self.ood_test_fold)
        self.ind_val_acc = self.get_predictor_accuracy("ind_val")
        self.ind_test_acc = self.get_predictor_accuracy("ind_test")

        if dsd_tnr+dsd_tpr <= 1:
            print("Using simple estimator since TPR+TNR<=1")
            estimator = SimpleEstimator

        ind_ndsd_acc = self.get_conditional_prediction_likelihood_estimates("ind_val", False)
        ind_dsd_acc = self.get_conditional_prediction_likelihood_estimates("ind_val", True)
        ood_ndsd_acc = self.get_conditional_prediction_likelihood_estimates(self.ood_val_folds, False)
        ood_dsd_acc = self.get_conditional_prediction_likelihood_estimates(self.ood_val_folds, True)

        # Stash these so the zero-OoD-accuracy ablation tree can reuse them
        # without rerunning the calibration.
        self.ind_ndsd_acc = ind_ndsd_acc
        self.ind_dsd_acc = ind_dsd_acc

        self.detector_tree = DetectorEventTree(dsd_tpr, dsd_tnr, ind_ndsd_acc, ind_dsd_acc, ood_ndsd_acc, ood_dsd_acc,
                                               estimator=estimator)
        self.base_tree = BaseEventTree(dsd_tpr=dsd_tpr, dsd_tnr=dsd_tnr, ood_acc=self.ood_val_acc,
                                       ind_acc=self.ind_val_acc, estimator=estimator)
        self.dsd_trace = Trace(trace_length)
        self.loss_trace_for_eval = Trace(trace_length)
        self.folds = self.df[self.df["ood"]]["fold"].unique()

    def get_conditional_prediction_likelihood_estimates(self, fold, monitor_verdict, num_samples=1000):
        if isinstance(fold, str):
            frame_copy = self.df[self.df["fold"] == fold]
        else:
            frame_copy = self.df[self.df["fold"].isin(list(fold))]
        samples = []
        for i in range(num_samples):
            sample = frame_copy.sample(replace=True).copy()
            sample["ood_pred"] = self.ood_detector.predict(sample)
            samples.append(sample)
        samples_df = pd.concat(samples)
        # get the likelihood of correct prediction given that the detectorr predicts monitor_verdict
        likelihood = samples_df[samples_df["ood_pred"] == monitor_verdict]["correct_prediction"].mean()

        if np.isnan(likelihood):
            return 0
        return likelihood

    def get_predictor_accuracy(self, fold):
        if isinstance(fold, str):
            return self.df[self.df["fold"] == fold]["correct_prediction"].mean()
        return self.df[self.df["fold"].isin(list(fold))]["correct_prediction"].mean()



    def sim(self, rate_groundtruth, num_batch_iters):
        self.detector_tree.rate = rate_groundtruth
        self.detector_tree.update_tree()
        has_shifted = self.detector_tree.rate_estimator.sample(num_batch_iters, rate_groundtruth)
        results = []
        results_trace = []
        for i in range(num_batch_iters):
            current_horizon_results = self.process(has_shifted, index=i)
            if current_horizon_results is not None:
                results.append(current_horizon_results)
                if len(results) > self.dsd_trace.trace_length:
                    df = pd.concat(
                        results[-self.dsd_trace.trace_length:])  # get the data corresponding to the last trace length
                    results_trace.append(df.groupby(["Tree"]).mean().reset_index())
        results_df = pd.concat(results_trace)
        results_df["Rate Error"] = np.abs(results_df["Estimated Rate"] - rate_groundtruth)
        results_df["Accuracy Error"] = np.abs(results_df["E[f(x)=y]"] - results_df["Accuracy"])
        return results_df


class UniformBatchSimulator(Simulator):
    """
    permits conditional data collection simulating model + ood detector
    """
    def __init__(self, df, ood_test_fold, ood_val_fold, estimator=ErrorAdjustmentEstimator, trace_length=100, **kwargs):
        super().__init__(df, ood_test_fold, ood_val_fold, estimator, trace_length, **kwargs)


    def sample_a_uniform_batch(self, fold):
        try:
            sample = self.df[self.df["fold"] == fold].sample()
        except ValueError:
            print("Error sampling from fold ", fold, "w/df: ", self.df.head(10))
            raise ValueError
        return sample

    def process(self, has_shifted, index):
        shifted = has_shifted[index]
        if shifted:
            batch = self.sample_a_uniform_batch(self.ood_test_fold)
        else:
            batch = self.sample_a_uniform_batch("ind_test")


        ood_pred = self.ood_detector.predict(batch)
        if isinstance(ood_pred, pd.Series):
            ood_pred = ood_pred.values[0] #dirty hack
        batch["ood_pred"] = ood_pred #todo, this is a hack
        # self.loss_trace_for_eval.update(batch["loss"])
        self.dsd_trace.update(int(ood_pred))

        if index>self.dsd_trace.trace_length: #update lambda after trace length

            self.detector_tree.update_rate(self.dsd_trace.trace)
            self.base_tree.update_rate(self.dsd_trace.trace)

            current_risk = self.detector_tree.calculate_risk(self.detector_tree.root)
            current_expected_accuracy = self.detector_tree.calculate_expected_accuracy(self.detector_tree.root)
            true_dsd_risk = self.detector_tree.get_true_risk_for_sample(batch)

            current_base_risk = self.base_tree.calculate_risk(self.base_tree.root)
            current_base_expected_accuracy = self.base_tree.calculate_expected_accuracy(self.base_tree.root)
            true_base_risk = self.base_tree.get_true_risk_for_sample(batch)
            accuracy = batch["correct_prediction"].mean()

            # PRE-0 ablation: same calibration, but assume OoD samples are
            # always wrong (ood_acc = 0). For both trees the leaves with
            # `corresponds_to_correct_prediction` that sit under the OoD
            # branch contribute zero, so E[f(x)=y] reduces to the InD branch:
            #   - BaseTree:     (1 - r) * ind_acc
            #   - DetectorTree: (1 - r) * [tnr * ind_ndsd_acc + (1-tnr) * ind_dsd_acc]
            tnr = self.detector_tree.dsd_tnr
            det_rate = self.detector_tree.rate
            base_rate = self.base_tree.rate
            current_expected_accuracy_zero_ood = (
                (1 - det_rate)
                * (tnr * self.ind_ndsd_acc + (1 - tnr) * self.ind_dsd_acc)
            )
            current_base_expected_accuracy_zero_ood = (1 - base_rate) * self.ind_val_acc

            data = pd.DataFrame( {"t":[index, index], "Tree": ["Detector Tree", "Base Tree"], "Risk Estimate": [current_risk, current_base_risk],
                                           "True Risk": [true_dsd_risk, true_base_risk], "E[f(x)=y]":[current_expected_accuracy, current_base_expected_accuracy],
                                           "E[f(x)=y]_zero_ood": [current_expected_accuracy_zero_ood, current_base_expected_accuracy_zero_ood],
                                           "Accuracy": [accuracy, accuracy], "ood_pred": [ood_pred, ood_pred], "is_ood": [shifted, shifted],
                                           "Estimated Rate":[self.detector_tree.rate, self.base_tree.rate],
                 "ind_acc": [self.ind_val_acc, self.ind_val_acc], "ood_val_acc": [self.ood_val_acc, self.ood_val_acc], "ood_test_acc": [self.ood_test_acc, self.ood_test_acc],
                                  "tpr": [self.detector_tree.dsd_tpr, self.detector_tree.dsd_tpr], "tnr": [self.detector_tree.dsd_tnr, self.detector_tree.dsd_tnr]})
            return data


    def sim(self, rate_groundtruth, num_batch_iters):
        self.detector_tree.rate = rate_groundtruth
        self.detector_tree.update_tree()
        has_shifted = self.detector_tree.rate_estimator.sample(num_batch_iters, rate_groundtruth)
        results = []
        results_trace = []
        for i in range(num_batch_iters):
            current_horizon_results = self.process(has_shifted, index=i)
            if current_horizon_results is not None:
                results.append(current_horizon_results)
                if len(results)>self.dsd_trace.trace_length:
                    df = pd.concat(results[-self.dsd_trace.trace_length:]) #get the data corresponding to the last trace length
                    results_trace.append(df.groupby(["Tree"]).mean().reset_index())
        results_df = pd.concat(results_trace)
        results_df["Rate Error"] = np.abs(results_df["Estimated Rate"] - rate_groundtruth)
        results_df["Accuracy Error"] = np.abs(results_df["E[f(x)=y]"] - results_df["Accuracy"])
        return results_df



if __name__ == '__main__':
    pass