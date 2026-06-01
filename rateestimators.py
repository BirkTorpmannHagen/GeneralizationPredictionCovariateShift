import numpy as np
from scipy.stats import beta
import math
class RateEstimator:
    def __init__(self):
        self.rate = 0.5

    def get_rate(self):
        return self.rate

    def update_tpr_tnr(self, tpr, tnr):
        self.tpr = float(tpr)
        self.tnr = float(tnr)

    def sample(self, n, rate_groundtruth):
        event = np.random.binomial(1, rate_groundtruth, n)
        return event

class ErrorAdjustmentEstimator(RateEstimator):
    def __init__(self, tpr=0.9, tnr=0.9):
        super().__init__()
        self.tpr = float(tpr)
        self.tnr = float(tnr)
        self.rate = 0.5  # last batch estimate (informational)

    def update(self, trace_list, return_ci=False, alpha=0.05, tol=1e-6):
        y = np.asarray(trace_list, dtype=float)
        n = len(y)
        if n == 0:
            return self.rate if not return_ci else (self.rate, (self.rate, self.rate))

        ybar = float(y.mean())
        J = self.tpr + self.tnr - 1.0
        if J <= 0:
            raise ValueError("TPR+TNR must exceed 1 (Youden's J > 0).")

        fpr = 1.0 - self.tnr
        if ybar < fpr - tol or ybar > self.tpr + tol:
            self.rate = ybar # drift detected
            # print("WARNING: tpr/tnr inconsistency detected, Setting rate to ybar.")
            return self.rate

        p_hat = (ybar - fpr) / J
        p_hat = float(np.clip(p_hat, 0.0, 1.0))
        self.rate = p_hat
        if not return_ci:
            return p_hat

        var_ybar = ybar * (1 - ybar) / n
        var_phat = var_ybar / (J * J)
        se = math.sqrt(var_phat)
        from scipy.stats import norm
        z = norm.ppf(1 - alpha / 2.0)
        lo, hi = p_hat - z * se, p_hat + z * se
        return p_hat, (max(0.0, lo), min(1.0, hi))

class SimpleEstimator(RateEstimator):
    def __init__(self, prior_rate=0.5):
        super().__init__()
        self.rate = prior_rate  # current rate estimate

    def update(self, trace_list):
        """Update the rate estimate based on the trace list."""
        trace = np.array(trace_list)
        self.rate = trace.mean()
        return self.rate


    def get_posterior_mean(self):
        """Return the mean of the posterior distribution."""
        return self.rate
