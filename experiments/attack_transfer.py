"""
Feasibility: does a white-box attack on ONE accuracy predictor transfer to the others?

SECURITY_FOLLOWUP.md hinges on whether the detectors fail independently or together.
The decoupling probe (decoupling_analysis.py) said "partially decoupled under benign
shift". This tests it under ATTACK, directly and cheaply — no soft-DR surrogate:

  * Every detector score is torch-differentiable w.r.t. the input (MSP/Energy/Entropy/
    GradNorm from logits; Mahalanobis/kNN/ViM/ReAct from the penultimate encoding). We
    reuse the FITTED detector params (mean/precision/subspace/reference bank) — constant
    w.r.t. the attack — and just re-enable gradients through the backbone.
  * For each detector i we run PGD to DRIVE ITS OOD-NESS BELOW THE ID THRESHOLD (make the
    monitor report "in-distribution / healthy") on genuinely shifted inputs.
  * We then read every other detector j on the SAME adversarial input.

Output: an N x N transfer matrix of post-attack evasion rates. Diagonal = self-evasion
(should be ~1). Off-diagonal high => attacking i also fools j => detectors are correlated
=> ensembling is a false sense of security. Off-diagonal low => decoupled => a joint
attack is needed => the follow-up's negative-transfer result is real.

Run: python -m experiments.attack_transfer            (writes figures/attack_transfer_*.csv)
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ooddetectors import FeatureSD
from features import _fit_maha, _fit_vim, _fit_react
import features as _F
from testbeds import (CCTTestBed, NICOTestBed, Office31TestBed, OfficeHomeTestBed,
                      Camelyon17TestBed, IWildCamTestBed)

FIGDIR = "figures"
TESTBEDS = {"CCT": CCTTestBed, "OfficeHome": OfficeHomeTestBed,
            "Office31": Office31TestBed, "NICO": NICOTestBed,
            "Camelyon17": Camelyon17TestBed, "IWildCam": IWildCamTestBed}
DETECTORS = ["msp", "energy", "entropy", "gradnorm", "mahalanobis", "knn", "vim", "react"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class DiffDetectors:
    """Differentiable, calibrated OOD-ness for every detector on a shared forward.

    NB: the classifier's get_encoding() is hardcoded torch.no_grad(), so we rebuild
    the backbone (all children except the final fc) ourselves to let gradients reach
    the input. logits = fc(enc) is exact for this ResNet head (get_encoding == fc input).
    """

    def __init__(self, clf, ref_np, knn_cap=5000):
        self.clf = clf.to(DEV).eval()
        for p in self.clf.parameters():
            p.requires_grad_(False)
        # Architecture-agnostic differentiable encoder + linear head.
        #  - ResNet: get_encoding() is hardcoded no_grad, so rebuild the backbone
        #    (all children except the final fc); logits = fc(enc) is exact.
        #  - ViT: get_encoding() IS differentiable (overridden, returns the post-LN
        #    CLS token = exact head input); logits = heads.head(enc).
        model = self.clf.model
        if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Linear):
            children = list(model.children())
            backbone = torch.nn.Sequential(*children[:-1]).to(DEV).eval()
            self.head = children[-1].to(DEV).eval()
            self._encode = lambda x: backbone(x).flatten(1)
        elif hasattr(model, "heads"):                      # torchvision ViT
            self.head = model.heads.head.to(DEV).eval()
            self._encode = lambda x: self.clf.get_encoding(x)
        else:
            raise ValueError("unsupported architecture for attack encoder")
        # fitted params (constant w.r.t. the attack)
        fm = _fit_maha(ref_np, clf)
        self.mu = fm["mu"].to(DEV)
        self.prec = fm["prec"].to(DEV)
        fv = _fit_vim(ref_np, clf)
        self.vmu = fv["mu"].to(DEV)
        self.principal = fv["principal"].to(DEV)
        self.alpha = fv["alpha"]
        self.react_c = _fit_react(ref_np, clf)["c"]
        R = np.asarray(ref_np)
        if R.ndim > 2:
            R = R.reshape(-1, R.shape[-1])
        if R.shape[0] > knn_cap:
            R = R[np.random.default_rng(0).choice(R.shape[0], knn_cap, replace=False)]
        self.knn_ref = torch.as_tensor(R, dtype=torch.float32, device=DEV)
        self.sign = {}       # calibrated so higher = more OOD
        self.thr = {}        # 95th-pct ind-val threshold on oriented score

    def _compute(self, x, which):
        """One backbone pass -> (logits, raw unoriented detector scores). Grad flows to x."""
        which = set(which) if which is not None else set(DETECTORS)
        z = self._encode(x)
        logits = F.linear(z, self.head.weight, self.head.bias)
        C = logits.shape[1]
        sm = F.softmax(logits, dim=1)
        p = sm.clamp_min(1e-12)
        out = {}
        if "msp" in which:
            out["msp"] = sm.max(dim=1).values
        if "atc" in which:
            # ATC-MC (Garg et al. 2022) score = max-softmax confidence; distinguished from
            # the MSP monitor by an ACCURACY-MATCHED threshold (set in calibration), not
            # a 95th-pct OOD threshold. Same differentiable signal as msp.
            out["atc"] = sm.max(dim=1).values
        if "atc_ne" in which:
            # ATC-NE (Garg et al. 2022) score = negative Shannon entropy (certainty);
            # stored as +entropy here (oriented sign=+1 => higher entropy = more OOD),
            # with an accuracy-matched threshold set in calibration.
            out["atc_ne"] = -(p * p.log()).sum(1)
        if "energy" in which:
            out["energy"] = torch.logsumexp(logits, dim=1)
        if "entropy" in which:
            out["entropy"] = -(p * p.log()).sum(1)
        if "gradnorm" in which:
            out["gradnorm"] = (sm - 1.0 / C).abs().sum(1) * z.abs().sum(1)  # h==enc for resnet head
        if "mahalanobis" in which:
            zc = z - self.mu
            out["mahalanobis"] = torch.einsum("bi,ij,bj->b", zc, self.prec, zc)
        if "knn" in which:
            out["knn"] = torch.cdist(z.float(), self.knn_ref).min(dim=1).values
        if "vim" in which:
            zcv = z - self.vmu
            proj = zcv @ self.principal
            res = (zcv - proj @ self.principal.T).norm(dim=1)
            out["vim"] = self.alpha * res - torch.logsumexp(logits, dim=1)
        if "react" in which:
            zr = torch.clamp(z, max=self.react_c)
            out["react"] = torch.logsumexp(F.linear(zr, self.head.weight, self.head.bias), dim=1)
        return logits, out

    def raw(self, x, which=None):
        return self._compute(x, which)[1]

    def oodness(self, x, which=None):
        _, raw = self._compute(x, which)
        return {k: self.sign.get(k, 1.0) * v for k, v in raw.items()}

    def forward_all(self, x, which=None):
        """(logits, oriented-oodness dict) in one backbone pass — for the joint attack."""
        logits, raw = self._compute(x, which)
        return logits, {k: self.sign.get(k, 1.0) * v for k, v in raw.items()}

    @torch.no_grad()
    def predict(self, x, mb=32):
        preds = []
        for c in x.split(mb):
            preds.append(F.linear(self._encode(c),
                                  self.head.weight, self.head.bias).argmax(1))
        return torch.cat(preds, 0)

    @torch.no_grad()
    def calibrate(self, ind_x, ood_x, q=0.95):
        ri = self.raw(ind_x)
        ro = self.raw(ood_x)
        self.scale = {}
        for k in DETECTORS:
            self.sign[k] = 1.0 if float(ro[k].mean()) > float(ri[k].mean()) else -1.0
            oriented = self.sign[k] * ri[k]
            self.thr[k] = float(torch.quantile(oriented, q))
            self.scale[k] = float(oriented.std()) + 1e-6   # to normalize the joint hinge


def pgd(det, x, target, eps, steps, step, mb=16):
    """PGD (L-inf) minimizing oodness[target] -> make detector `target` say ID.

    Chunked into mini-batches so peak memory is one chunk's forward+backward graph
    (a full ResNet-50 backward on the whole batch OOMs alongside other GPU jobs).
    """
    outs = []
    for chunk in x.split(mb):
        x0 = chunk.detach()
        delta = torch.zeros_like(x0, requires_grad=True)
        for _ in range(steps):
            s = det.oodness(x0 + delta, which=[target])[target].sum()
            g, = torch.autograd.grad(s, delta)
            with torch.no_grad():
                delta -= step * g.sign()          # descend oodness
                delta.clamp_(-eps, eps)
            delta.requires_grad_(True)
        outs.append((x0 + delta).detach())
    return torch.cat(outs, 0)


def _grab(loader_dict, key, n, dev):
    xs, ys = [], []
    for x, y, *_ in loader_dict[key]:
        xs.append(x)
        ys.append(y)
        if sum(t.shape[0] for t in xs) >= n:
            break
    x = torch.cat(xs, 0)[:n].to(dev)
    y = torch.cat(ys, 0)[:n].to(dev)
    return x, y


def run(dataset="CCT", model="resnet", mode="noise", n=48, eps=0.5, steps=40, step=0.05):
    bench = TESTBEDS[dataset](model=model, mode=mode, batch_size=32)
    fsd = FeatureSD(bench.classifier, [])
    fsd.register_testbed(bench)
    ref_np = fsd.get_encodings(fsd._capped_reference_loader())
    det = DiffDetectors(bench.classifier, ref_np)

    ind_x, _ = _grab(bench.ind_val_loader(), "ind_val", n, DEV)
    ood_loaders = bench.ood_loaders()
    strong_key = list(ood_loaders.keys())[-1]           # highest shift intensity
    ood_x, _ = _grab(ood_loaders, strong_key, n, DEV)
    det.calibrate(ind_x, ood_x)

    # clean OOD-ness + flag rates on the shifted batch
    with torch.no_grad():
        clean = {k: v.detach() for k, v in det.oodness(ood_x).items()}
    clean_flag = {k: float((clean[k] > det.thr[k]).float().mean()) for k in DETECTORS}

    rows = []
    trans = pd.DataFrame(index=DETECTORS, columns=DETECTORS, dtype=float)
    for i in DETECTORS:
        x_adv = pgd(det, ood_x, i, eps, steps, step)
        with torch.no_grad():
            adv = det.oodness(x_adv)
        for j in DETECTORS:
            evaded = float((adv[j] <= det.thr[j]).float().mean())   # fraction j now says ID
            trans.loc[i, j] = evaded
            # oodness z-reduction relative to clean spread
            spread = float(clean[j].std()) + 1e-6
            zred = float((clean[j].mean() - adv[j].mean()) / spread)
            rows.append({"attack": i, "readout": j, "evade_rate": evaded,
                         "clean_flag": clean_flag[j], "z_reduction": zred})
    detail = pd.DataFrame(rows)
    os.makedirs(FIGDIR, exist_ok=True)
    tag = f"{dataset}_{model}_{mode}"
    detail.to_csv(f"{FIGDIR}/attack_transfer_{tag}.csv", index=False)
    trans.to_csv(f"{FIGDIR}/attack_transfer_matrix_{tag}.csv")

    print(f"\n=== {tag}: clean OOD flag rate on shifted batch ===")
    print({k: round(v, 2) for k, v in clean_flag.items()})
    print(f"\n=== Transfer matrix: P(readout j says ID | attack on i)  [rows=attack, cols=readout] ===")
    print(trans.astype(float).round(2).to_string())
    diag = np.diag(trans.values.astype(float))
    off = trans.values.astype(float)[~np.eye(len(DETECTORS), dtype=bool)]
    print(f"\nmean self-evasion (diagonal) = {diag.mean():.2f}")
    print(f"mean cross-evasion (off-diagonal) = {off.mean():.2f}")
    # confidence-family vs feature-family transfer
    conf = ["msp", "energy", "entropy", "gradnorm", "react"]
    feat = ["mahalanobis", "knn", "vim"]
    cf = trans.loc[conf, feat].values.astype(float).mean()
    fc = trans.loc[feat, conf].values.astype(float).mean()
    print(f"confidence-attack -> feature-readout evasion = {cf:.2f}")
    print(f"feature-attack -> confidence-readout evasion = {fc:.2f}")
    print("\nLow off-diagonal / low cross-family => DECOUPLED => joint attack needed"
          " => follow-up is real. High => correlated => ensemble is false security.")
    return detail, trans


def joint_attack(det, x, y_true, eps, steps, step, targets=None, mb=16,
                 margin=0.5, lam_cls=1.0):
    """Adaptive joint attack (Carlini/Tramer bar): simultaneously drive EVERY
    detector in `targets` below its ID threshold AND keep the model confidently
    WRONG (CE to the nearest wrong class). Realizes the threat model: true accuracy
    low while the monitor reports healthy.
    """
    targets = list(targets) if targets is not None else list(DETECTORS)
    outs = []
    for cx, cy in zip(x.split(mb), y_true.split(mb)):
        x0 = cx.detach()
        with torch.no_grad():
            lg, _ = det.forward_all(x0, which=[])
            top2 = lg.topk(2, dim=1).indices
            pred = top2[:, 0]
            y_adv = torch.where(pred == cy, top2[:, 1], pred)   # nearest wrong class
        delta = torch.zeros_like(x0, requires_grad=True)
        for _ in range(steps):
            logits, oo = det.forward_all(x0 + delta, which=targets)
            pen = 0.0
            for k in targets:
                pen = pen + F.relu((oo[k] - det.thr[k]) / det.scale[k] + margin).mean()
            loss = pen + lam_cls * F.cross_entropy(logits, y_adv)
            g, = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= step * g.sign()
                delta.clamp_(-eps, eps)
            delta.requires_grad_(True)
        outs.append((x0 + delta).detach())
    return torch.cat(outs, 0)


@torch.no_grad()
def _report(det, x, y_true):
    pred = det.predict(x)
    oo = {k: v for k, v in det.oodness(x).items()}
    flag = {k: (oo[k] > det.thr[k]) for k in DETECTORS}
    any_flag = torch.stack([flag[k] for k in DETECTORS], 0).any(0)   # ensemble catches
    all_id = ~any_flag                                               # ensemble fooled
    wrong = pred != y_true
    return {
        "acc": float((~wrong).float().mean()),
        "per_detector_evasion": {k: float((~flag[k]).float().mean()) for k in DETECTORS},
        "ensemble_evasion_all_ID": float(all_id.float().mean()),
        "misclassified": float(wrong.float().mean()),
        "SUCCESS_wrong_and_healthy": float((wrong & all_id).float().mean()),
    }


def run_joint(dataset="CCT", model="resnet", mode="noise", n=64,
              eps_list=(0.1, 0.25, 0.5, 1.0), steps=60):
    bench = TESTBEDS[dataset](model=model, mode=mode, batch_size=32)
    fsd = FeatureSD(bench.classifier, [])
    fsd.register_testbed(bench)
    ref_np = fsd.get_encodings(fsd._capped_reference_loader())
    det = DiffDetectors(bench.classifier, ref_np)

    ind_x, _ = _grab(bench.ind_val_loader(), "ind_val", n, DEV)
    ood_loaders = bench.ood_loaders()
    strong_key = list(ood_loaders.keys())[-1]
    ood_x, ood_y = _grab(ood_loaders, strong_key, n, DEV)
    det.calibrate(ind_x, ood_x)

    tag = f"{dataset}_{model}_{mode}"
    clean = _report(det, ood_x, ood_y)
    print(f"\n=== {tag} | clean shifted batch (n={n}, shift={strong_key}) ===")
    print(f"  true accuracy={clean['acc']:.2f}  ensemble-fooled(all-ID)="
          f"{clean['ensemble_evasion_all_ID']:.2f}  misclassified={clean['misclassified']:.2f}")

    conf = ["msp", "energy", "entropy"]
    rows = []
    for eps in eps_list:
        step = 2.5 * eps / steps
        for name, targets in [("confidence-only", conf), ("joint-ALL", DETECTORS)]:
            x_adv = joint_attack(det, ood_x, ood_y, eps, steps, step, targets=targets)
            r = _report(det, x_adv, ood_y)
            rows.append({"tag": tag, "attack": name, "eps": eps,
                         "misclassified": r["misclassified"],
                         "ensemble_evasion_all_ID": r["ensemble_evasion_all_ID"],
                         "SUCCESS_wrong_and_healthy": r["SUCCESS_wrong_and_healthy"],
                         **{f"evade_{k}": r["per_detector_evasion"][k] for k in DETECTORS}})
            print(f"  eps={eps:<4} {name:15s} misclass={r['misclassified']:.2f} "
                  f"all-ID={r['ensemble_evasion_all_ID']:.2f} "
                  f"SUCCESS(wrong&healthy)={r['SUCCESS_wrong_and_healthy']:.2f}")
    out = pd.DataFrame(rows)
    os.makedirs(FIGDIR, exist_ok=True)
    out.to_csv(f"{FIGDIR}/attack_joint_{tag}.csv", index=False)
    print("\nSUCCESS = fraction of shifted inputs that are BOTH misclassified AND evade"
          " the FULL ensemble. High only for joint-ALL at large eps => the ensemble"
          " forces a hard, high-budget adaptive attack (Carlini/Tramer bar).")
    return out


def _cw_keepwrong(logits, y, kappa):
    """CW hinge, 0 once the sample is misclassified with margin kappa."""
    lyt = logits.gather(1, y.view(-1, 1)).squeeze(1)
    other = logits.clone()
    other.scatter_(1, y.view(-1, 1), float("-inf"))
    max_other = other.max(1).values
    return F.relu(lyt - max_other + kappa)          # push true logit below best-other by kappa


def _succeeded(det, x, y):
    """Per-sample (misclassified AND every detector below its ID threshold)."""
    logits, oo = det.forward_all(x)
    wrong = logits.argmax(1) != y
    all_id = torch.stack([oo[k] <= det.thr[k] for k in DETECTORS], 0).all(0)
    return (wrong & all_id)


def tuned_attack(det, x, y_true, eps, targets=None, steps=250, restarts=6,
                 lr=0.05, margin=0.1, kappa=5.0, w_cls=1.0, mb=16, clamp_valid=True):
    """Best-effort adaptive joint attack: Adam + tanh box-reparam + CW-margin losses
    + Lagrangian dual-ascent per-detector weights + random restarts. Returns the
    best (success-maximising) adversarial input per sample.
    """
    targets = list(targets) if targets is not None else list(DETECTORS)
    outs = []
    for cx, cy in zip(x.split(mb), y_true.split(mb)):
        x0 = cx.detach()
        best_adv = x0.clone()
        best_score = None            # continuous, target-aware "how good is this adv"
        for r in range(restarts):
            w = (torch.zeros_like(x0) if r == 0
                 else 0.1 * torch.randn_like(x0)).requires_grad_(True)
            opt = torch.optim.Adam([w], lr=lr)
            lam = {k: 1.0 for k in targets}
            for t in range(steps):
                # L-inf eps ball; optionally project to a valid image ([0,1], since
                # transforms are Resize+ToTensor with NO normalization).
                x_adv = x0 + eps * torch.tanh(w)
                if clamp_valid:
                    x_adv = torch.clamp(x_adv, 0.0, 1.0)
                logits, oo = det.forward_all(x_adv, which=targets)
                evade = 0.0
                for k in targets:
                    evade = evade + lam[k] * F.relu(
                        (oo[k] - det.thr[k]) / det.scale[k] + margin).mean()
                loss = w_cls * _cw_keepwrong(logits, cy, kappa).mean() + evade
                opt.zero_grad()
                loss.backward()
                opt.step()
                if (t + 1) % 25 == 0:                # dual ascent: focus stubborn detectors
                    with torch.no_grad():
                        for k in targets:
                            viol = float(((oo[k] > det.thr[k]).float().mean()))
                            if viol > 0.3:
                                lam[k] = min(lam[k] * 1.5, 50.0)
            with torch.no_grad():
                adv = x0 + eps * torch.tanh(w)
                if clamp_valid:                       # match the optimization constraint
                    adv = torch.clamp(adv, 0.0, 1.0)
                adv = adv.detach()
                # Target-aware, CONTINUOUS acceptance: keep the perturbation that, while
                # keeping the model wrong, evades the MOST detectors IN THE ATTACKED
                # monitor (not the full 8). Continuous so a partial evasion still beats
                # clean — never silently resets best_adv to the unperturbed image.
                logits, oo = det.forward_all(adv, which=targets)
                wrong = (logits.argmax(1) != cy).float()
                n_id = torch.stack([(oo[k] <= det.thr[k]).float() for k in targets], 0).sum(0)
                score = wrong * (n_id + 1.0)          # wrong AND evade as many targets as possible
                if best_score is None:
                    best_score, best_adv = score.clone(), adv.clone()
                else:
                    better = score > best_score
                    best_adv[better] = adv[better]
                    best_score = torch.maximum(best_score, score)
        outs.append(best_adv)
    return torch.cat(outs, 0)


def run_tuned(dataset="CCT", model="resnet", mode="noise", n=64,
              eps_list=(8/255, 16/255, 32/255, 64/255), steps=200, restarts=5,
              verbose=True):
    """Tuned adaptive attack under a PROPER threat model: L-inf eps in [0,1] pixel
    units (inputs are un-normalized), with valid-image projection. eps=8/255 is the
    standard imperceptible budget."""
    bench = TESTBEDS[dataset](model=model, mode=mode, batch_size=32)
    fsd = FeatureSD(bench.classifier, [])
    fsd.register_testbed(bench)
    ref_np = fsd.get_encodings(fsd._capped_reference_loader())
    det = DiffDetectors(bench.classifier, ref_np)
    ind_x, _ = _grab(bench.ind_val_loader(), "ind_val", n, DEV)
    ood_loaders = bench.ood_loaders()
    strong_key = list(ood_loaders.keys())[-1]
    ood_x, ood_y = _grab(ood_loaders, strong_key, n, DEV)
    det.calibrate(ind_x, ood_x)

    tag = f"{dataset}_{model}_{mode}"
    clean = _report(det, ood_x, ood_y)
    if verbose:
        print(f"\n=== TUNED adaptive attack (valid-image, pixel-eps) | {tag} "
              f"(n={n}, shift={strong_key}) ===", flush=True)
        print(f"  clean: acc={clean['acc']:.2f} misclassified={clean['misclassified']:.2f} "
              f"all-ID={clean['ensemble_evasion_all_ID']:.2f}", flush=True)
    rows = []
    for eps in eps_list:
        x_adv = tuned_attack(det, ood_x, ood_y, eps, targets=DETECTORS,
                             steps=steps, restarts=restarts)
        r = _report(det, x_adv, ood_y)
        rows.append({"tag": tag, "eps": eps, "eps_255": round(eps * 255),
                     "misclassified": r["misclassified"],
                     "ensemble_evasion_all_ID": r["ensemble_evasion_all_ID"],
                     "SUCCESS_wrong_and_healthy": r["SUCCESS_wrong_and_healthy"],
                     **{f"evade_{k}": r["per_detector_evasion"][k] for k in DETECTORS}})
        if verbose:
            hard = sorted(((r["per_detector_evasion"][k], k) for k in DETECTORS))[:3]
            print(f"  eps={round(eps*255)}/255  SUCCESS(wrong&healthy)="
                  f"{r['SUCCESS_wrong_and_healthy']:.2f} all-ID="
                  f"{r['ensemble_evasion_all_ID']:.2f} misclass={r['misclassified']:.2f} "
                  f"| hardest: {[(k, round(v,2)) for v,k in hard]}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{FIGDIR}/attack_tuned_{tag}.csv", index=False)
    if verbose:
        print("\nThreat model matters: eps=8/255 is imperceptible. SUCCESS~0 there means the"
              " ensemble resists imperceptible adaptive attacks; SUCCESS rising only at large"
              " eps means it is defeatable only by visible perturbations.", flush=True)
    return out


@torch.no_grad()
def _evade_and_success(det, x, y, d):
    """(fraction d says ID, fraction wrong AND d says ID) after an attack."""
    logits, oo = det.forward_all(x, which=[d])
    wrong = logits.argmax(1) != y
    id_d = oo[d] <= det.thr[d]
    return float(id_d.float().mean()), float((wrong & id_d).float().mean())


def run_single_vs_ensemble(dataset="CCT", model="resnet", mode="noise",
                           eps=8/255, n=48, steps=150, restarts=4):
    """The 'use ensembles' linchpin: at a FIXED valid-image budget, can each detector
    be fooled INDIVIDUALLY (targeted attack) while the FULL ensemble cannot be fooled
    jointly? If single-detector evasion is high but ensemble all-ID ~ 0, diversity is
    what buys the robustness."""
    bench = TESTBEDS[dataset](model=model, mode=mode, batch_size=32)
    fsd = FeatureSD(bench.classifier, []); fsd.register_testbed(bench)
    ref = fsd.get_encodings(fsd._capped_reference_loader())
    det = DiffDetectors(bench.classifier, ref)
    ind, _ = _grab(bench.ind_val_loader(), "ind_val", n, DEV)
    ol = bench.ood_loaders(); k = list(ol.keys())[-1]
    x, y = _grab(ol, k, n, DEV); det.calibrate(ind, _grab(ol, k, n, DEV)[0])
    tag = f"{dataset}_{model}_{mode}_eps{round(eps*255)}"
    print(f"\n=== single-vs-ensemble @ eps={round(eps*255)}/255 | {tag} (shift={k}) ===", flush=True)
    rows = []
    for d in DETECTORS:                                   # attack ONE detector at a time
        xa = tuned_attack(det, x, y, eps, targets=[d], steps=steps, restarts=restarts,
                          clamp_valid=True)
        ev, succ = _evade_and_success(det, xa, y, d)
        rows.append({"target": d, "self_evaded": ev, "single_success": succ})
        print(f"  attack[{d:12}]  self-evaded={ev:.2f}  single-success(wrong&ID)={succ:.2f}", flush=True)
    xa_all = tuned_attack(det, x, y, eps, targets=DETECTORS, steps=steps,
                          restarts=restarts, clamp_valid=True)
    r_all = _report(det, xa_all, y)
    ens = r_all["ensemble_evasion_all_ID"]
    print(f"  ENSEMBLE (attack all 8 jointly): all-ID={ens:.2f} "
          f"SUCCESS(wrong&healthy)={r_all['SUCCESS_wrong_and_healthy']:.2f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{FIGDIR}/attack_single_vs_ensemble_{tag}.csv", index=False)
    mean_single = out["self_evaded"].mean()
    print(f"\n  => mean single-detector self-evasion={mean_single:.2f} vs ensemble all-ID={ens:.2f}."
          f" If single is high and ensemble ~0, DIVERSITY buys robustness (use ensembles).",
          flush=True)
    return out, ens


if __name__ == "__main__":
    run_single_vs_ensemble()
