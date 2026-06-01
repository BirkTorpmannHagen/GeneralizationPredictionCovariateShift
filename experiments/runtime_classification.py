import os.path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pygam import LinearGAM
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from scipy.stats import spearmanr
from multiprocessing import Pool, cpu_count
import seaborn as sns
import matplotlib.pyplot as plt
from components import EnsembleOODDetector, LogisticRiskCalibrator
from itertools import product
from rateestimators import ErrorAdjustmentEstimator
from simulations import UniformBatchSimulator
from utils import *



from itertools import product
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def parallel_compute_ood_detector_prediction_accuracy(data_filtered, threshold_method, dataset, feature):
    data_dict = []
    ood_val_folds = data_filtered[(data_filtered["Organic"] == True) & (data_filtered["ood"] == True)]["fold"].unique()
    assert data_filtered["Model"].nunique() == 1, "Data contains multiple models"
    for ind_val_fold, ind_test_fold in product(["ind_val", "ind_test"], repeat=2):
        for ood_val_fold in ood_val_folds:
            data_copy = data_filtered.copy()
            if ood_val_fold in ["train", "ind_val", "ind_test"] or ood_val_fold.split("_")[0] in SYNTHETIC_SHIFTS:
                # dont calibrate on ind data or synthetic ood data
                continue

            data_train = data_copy[
                (data_copy["fold"] == ood_val_fold) | (data_copy["fold"] == ind_val_fold)
            ].copy()

            dsd = OODDetector(data_train, threshold_method=threshold_method)
            for fold in data_copy["fold"].unique():
                if fold in ["train"]:
                    continue  # not ood

                data_test = data_copy[(data_copy["fold"] == fold)].copy()

                shift = fold.split("_")[0]  if "_" in fold else "Organic"
                shift_intensity = fold.split("_")[-1] if "_" in fold else "Organic"
                dr  = dsd.get_dr(data_test)
                # if ba<0.5:
                #     print(f"Warning: BA<<0.5 for {data_test['Model'].unique()} {dataset} {feature} {threshold_method} OOD Val: {ood_val_fold} OOD Test: {ood_test_fold} InD Val: {ind_val_fold} InD Test: {ind_test_fold} BA: {ba},  {dsd.threshold}")
                #     dsd.plot_hist()


                data_dict.append({
                    "Dataset": dataset,
                    "feature_name": feature,
                    "Threshold Method": threshold_method,
                    "OoD Val Fold": ood_val_fold,
                    "InD Val Fold": ind_val_fold,
                    "Fold": fold,
                    "Shift": shift,
                    "Shift Intensity": shift_intensity,
                    "DR": dr,
                    "Accuracy": data_test["correct_prediction"].mean(),
                })
    return data_dict


def _compute_one(args):
    data_filtered, threshold_method, dataset, feature = args
    try:
        return parallel_compute_ood_detector_prediction_accuracy(
            data_filtered, threshold_method, dataset, feature
        )
    except Exception as e:
        return {
            "Dataset": dataset,
            "feature_name": feature,
            "Threshold Method": threshold_method,
            "error": str(e),
        }


def _make_jobs_for_dataset_feature(df, dataset, feature):
    data_filtered = df[(df["Dataset"] == dataset) & (df["feature_name"] == feature)]
    if data_filtered.empty:
        return []
    return [(data_filtered, tm, dataset, feature) for tm in THRESHOLD_METHODS]


# ---- main entry ----
def ood_detector_correctness_prediction_accuracy(batch_size, model="resnet", shift=""):
    prefix = "data"
    df = load_all(batch_size=batch_size, shift=shift, samples=100)
    df = df[df["Model"] == model]
    df = df[df["fold"] != "train"]
    if df.empty:
        print("No data loaded.")
        return

    # build ALL jobs across datasets and features (parallelized at this level too)
    all_jobs = []
    for dataset in DATASETS:
        for feature in DSDS:
            all_jobs.extend(_make_jobs_for_dataset_feature(df, dataset, feature))

    total_jobs = len(all_jobs)
    if total_jobs == 0:
        print("No jobs to run.")
        return

    results_all = []
    n_procs = max(1, cpu_count() - 1)

    with Pool(processes=n_procs) as pool, tqdm(total=total_jobs, desc="Computing") as pbar:
        for out in pool.imap_unordered(_compute_one, all_jobs, chunksize=1):
            pbar.update(1)
            if out is None:
                continue
            results_all.append(out)

    if not results_all:
        print("No results produced.")
        return

    flat_success = [row for out in results_all if isinstance(out, list) for row in out]
    flat_errors = [out for out in results_all if isinstance(out, dict) and "error" in out]

    data = pd.DataFrame(flat_success)
    if not data.empty:
        data["feature_name"].replace(DSD_PRINT_LUT, inplace=True)
    os.mkdir(f"{prefix}/{model}/ood_detector_data") if not os.path.exists(f"{prefix}/{model}/ood_detector_data") else None
    if flat_errors:
        pd.DataFrame(flat_errors).to_csv(
            f"{prefix}/{model}/ood_detector_data/ood_detector_errors_{batch_size}.csv",
            index=False
        )

    for dataset in DATASETS:
        data_ds = data[data["Dataset"] == dataset]
        if data_ds.empty:
            continue
        data_ds.to_csv(
            f"{prefix}/{model}/ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv",
            index=False
        )

def get_all_ood_detector_data(batch_size, filter_organic=False, filter_best=False, threshold_method="val_optimal"):
    dfs = []
    prefix = "data"
    for model in MODELS:
        if not os.path.exists(f"{prefix}/{model}/ood_detector_data"):
            os.makedirs(f"{prefix}/{model}/ood_detector_data")
        # Determine which datasets are expected for this model.
        expected_datasets = (
            ["Polyp"] if model in SEG_MODELS else [d for d in DATASETS if d != "Polyp"]
        )
        missing = any(
            not os.path.exists(f"{prefix}/{model}/ood_detector_data/ood_detector_correctness_{d}_{batch_size}.csv")
            for d in expected_datasets
        )
        if missing:
            ood_detector_correctness_prediction_accuracy(batch_size, model=model, shift="")
        for dataset, feature in itertools.product(DATASETS, DSDS):
            if dataset=="Polyp" and model not in SEG_MODELS:
                continue
            if dataset!="Polyp" and model in SEG_MODELS:
                continue
            try:
                df = pd.read_csv(f"{prefix}/{model}/ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv")
            except FileNotFoundError:
                print(f"{prefix}/{model}/ood_detector_data/ood_detector_correctness_{dataset}_{batch_size}.csv not found")
                continue
            df["Model"] = model
            dfs.append(df)
    df = pd.concat(dfs)
    print(df.columns)
    df = df.groupby(["Dataset", "feature_name", "Threshold Method", "Fold", "Shift", "Shift Intensity", "Model"])[["DR", "Accuracy"]].mean(numeric_only=True).reset_index()

    if "OOD==f(x)=y" in df.columns:
        df = df[df["OOD==f(x)=y"]==False]
    if "Performance Calibrated" in df.columns:
        df = df[df["Performance Calibrated"]==True]
    if "Threshold Method" in df.columns:
        df = df[df["Threshold Method"] == threshold_method]

    if filter_organic:
        df = df[df["Shift Intensity"] == "Organic"]


    if filter_best:
        if df.empty:
            return df
        #filter for best models and features per dataset
        df_organic = df[df["Shift Intensity"] == "Organic"] #just consider the organic data for best selection

        def tpr_acc_corr(g):
            # guard against degenerate groups
            if len(g) < 2:
                return np.nan
            if g["DR"].nunique() <= 1 or g["Accuracy"].nunique() <= 1:
                return np.nan
            return g["DR"].corr(g["Accuracy"])

        corrs = (
            df
            .groupby(["Dataset", "Model", "feature_name"])
            .apply(tpr_acc_corr)
        )
        # Robust to pandas versions where .apply may return Series or DataFrame:
        if isinstance(corrs, pd.DataFrame):
            corrs = corrs.iloc[:, 0]
        corrs = corrs.rename("tpr_acc_corr").reset_index()

        # strongest correlation = largest absolute value
        corrs["abs_corr"] = corrs["tpr_acc_corr"].abs()

        # pick the (Model, feature_name) with strongest |corr| per Dataset
        best_corr = corrs.loc[
            corrs.groupby("Dataset")["abs_corr"].idxmax()
        ]

        # filter original df to only those best (Dataset, Model, feature_name)
        df = df.merge(
            best_corr[["Dataset", "Model", "feature_name"]],
            on=["Dataset", "Model", "feature_name"],
            how="inner",
        )


        # meaned_ba = df_organic.groupby(["Dataset", "Model", "feature_name"])["ba"].mean().reset_index() #
        #
        # best_ba = meaned_ba.loc[meaned_ba.groupby(["Dataset"])["ba"].idxmax()]
        # df = df.merge(best_ba[["Dataset", "Model", "feature_name"]], on=["Dataset", "Model", "feature_name"], how="inner")

    return df


