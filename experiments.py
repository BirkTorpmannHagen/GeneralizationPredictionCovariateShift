import warnings

from experiments.pra import collect_re_accuracy_estimation_data

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")  # non-interactive: write PDFs, don't open windows
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False  # local TeX install lacks cmss10.tfm; mathtext is fine for figure prep
from experiments.accuracy_prediction import *
np.set_printoptions(precision=3, suppress=True)

def run_acc_prediction_experiments():
    sequence_length_sensitivity([2, 4, 8, 16, 32, 64, 128, 256, 512], n_samples=50, error="std")
    method_statistical_tests()
    test_generalization_gap_estimation(1, pretrain=True)

    dr_gap_correlation_distribution(1, pretrain=True)

    test_generalization_gap_estimation(1, pretrain=True)
    dr_gap_correlation_distribution(1, pretrain=True)
    threshold_method_comparison(1, pretrain=True)
    paired, stats = method_statistical_tests(batch_size=1, pretrain=True)

    pivot, corrected_rows = accuracy_prediction_table(
        batch_size=1,
        pretrain=True,
    )

    pivot.to_csv("figures/corrected_shift_type_loo_comparison.csv")
    corrected_rows.to_csv("figures/shift_type_loo_rows.csv", index=False)
    #
    print("\n=== Leave-one-shift-type-out MAE ===")
    print(pivot.round(4).to_string())

    predicted_vs_true_gap_grid(corrected_rows)
    error_per_accuracy(corrected_rows)
    intensity_breakdown_plot(corrected_rows)

    # ood_detector_ba()

if __name__ == '__main__':
    run_acc_prediction_experiments()
