import itertools
import os.path
from multiprocessing import Pool

import seaborn as sns
from matplotlib import pyplot as plt

from experiments.runtime_classification import get_all_ood_detector_data
from rateestimators import ErrorAdjustmentEstimator
from simulations import UniformBatchSimulator
from utils import *

pd.set_option("display.precision", 3)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering


def simulate_dsd_accuracy_estimation(data, rate, val_set, test_fold, feature_name):
    sim = UniformBatchSimulator(data, ood_test_fold=test_fold, ood_val_fold=val_set,
                                estimator=ErrorAdjustmentEstimator)
    results = sim.sim(rate, 600)
    results = results.groupby(["Tree"]).mean().reset_index()
    results["dsd"] = feature_name
    results["rate"] = rate
    results["test_set"] = test_fold
    # When calibrated on multiple folds (leave-one-shift-type-out protocol),
    # `val_set` is a list and cannot be broadcast into a column without care;
    # store a stable string identifier and let the caller overwrite if needed.
    if isinstance(val_set, str):
        results["val_set"] = val_set
    else:
        results["val_set"] = ",".join(map(str, val_set))
    return results














def _shift_type_of_fold(fold):
    fold = str(fold)
    if fold in ("train", "ind_val", "ind_test"):
        return fold
    if "_" in fold:
        return fold.split("_")[0]
    return "Organic"


def collect_re_accuracy_estimation_data(pretrain=True):
    """
    Leave-one-synthetic-shift-type-out PRE calibration.

    For each synthetic shift type S in a dataset, calibrate on every synthetic
    fold whose shift type != S, then evaluate on (a) every fold in S and
    (b) every organic OoD fold. Organic folds are never used for calibration.
    This mirrors the protocol used by the regression-based "Ours" method, so
    PRE no longer gets the unfair advantage of seeing organic folds during
    calibration.
    """
    bins = 11
    prefix = "data/pretrain" if pretrain else "data/nopretrain"

    best = get_all_ood_detector_data(batch_size=1, filter_organic=False, filter_best=True, pretrain=pretrain)

    for dataset in best["Dataset"].unique()[::-1]:

        assert best[best["Dataset"] == dataset]["Model"].unique().shape[0] == 1, "Multiple models found for dataset"
        model = best[best["Dataset"] == dataset]["Model"].unique()[0]
        dsd_accuracies = best[(best["Dataset"] == dataset)&(best["Model"]==model)]
        if dsd_accuracies.empty:
            continue
        dfs = []
        feature_name = dsd_accuracies["feature_name"].values[0]

        data = load_data(dataset, DSD_LUT[feature_name], batch_size=1, samples=1000, shift="",
                         model=model, pretrain=pretrain)
        if data.empty:
            continue


        all_folds = data["fold"].unique().tolist()

        excluded_shift_types = {"ind"}
        synthetic_folds = [
            f for f in all_folds
            if _shift_type_of_fold(f) in SYNTHETIC_SHIFTS
        ]
        organic_ood_folds = [
            f for f in all_folds
            if (f not in ("train", "ind_val", "ind_test"))
            and (_shift_type_of_fold(f) not in SYNTHETIC_SHIFTS)
            and (_shift_type_of_fold(f) not in excluded_shift_types)
        ]
        synthetic_shift_types = sorted({_shift_type_of_fold(f) for f in synthetic_folds})

        if len(synthetic_shift_types) < 2:
            print(f"[PRE] {dataset}: <2 synthetic shift types available, skipping")
            continue

        with tqdm(total=bins * (len(synthetic_folds)
                                + len(synthetic_shift_types) * len(organic_ood_folds))) as pbar:
            for held_shift_type in synthetic_shift_types:
                # Calibrate on every synthetic fold whose shift type is NOT the
                # held-out one. Organic folds are never used for calibration —
                # this matches the leave-one-synthetic-shift-type-out protocol
                # used by the regression-based "Ours" method.
                calibration_folds = [
                    f for f in synthetic_folds
                    if _shift_type_of_fold(f) != held_shift_type
                ]
                if not calibration_folds:
                    continue

                test_folds = [
                    f for f in synthetic_folds
                    if _shift_type_of_fold(f) == held_shift_type
                ] + list(organic_ood_folds)

                for test_fold in test_folds:
                    print(f"{dataset}-{model}-heldout:{held_shift_type}-test:{test_fold}")

                    pool = Pool(bins)
                    results = pool.starmap(simulate_dsd_accuracy_estimation, [
                        (data, rate, calibration_folds, test_fold, feature_name) for rate
                        in np.linspace(0, 1, bins)])
                    pool.close()

                    for result in results:
                        result["Model"] = model
                        # `heldout_shift_type` = the synthetic shift type we
                        # excluded from calibration (and that this row's
                        # synthetic test_fold belongs to, when test_fold is
                        # synthetic). `val_set` (set inside
                        # `simulate_dsd_accuracy_estimation`) keeps the
                        # comma-joined list of calibration folds — i.e. the
                        # shifts the detector was *actually* calibrated on.
                        result["heldout_shift_type"] = held_shift_type
                        dfs.append(result)
                        pbar.update(1)
        df_final = pd.concat(dfs)
        print(df_final.head(10))
        out_dir = f"{prefix}/{model}/pra_data/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        df_final.to_csv(f"{out_dir}{dataset}_pre_results.csv")
















