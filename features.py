import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


# ---------------------------------------------------------------------------
# Shared per-batch forward. All post-hoc detectors run on the same batch, so we
# compute the backbone forward ONCE (capturing the true classifier-head input via
# a forward hook, which is exact for both ResNet and ViT) and reuse it, instead
# of ~8 redundant forwards per batch. Keyed by tensor identity (not id()) to be
# safe against id reuse across batches.
# ---------------------------------------------------------------------------
_SHARED = {"x": None, "logits": None, "h": None, "enc": None}


def _shared(model, x, need_enc=False):
    if _SHARED["x"] is not x:
        cap = {}
        handle = None
        try:
            lin = _final_linear(model)

            def _hook(m, inp, out):
                cap["h"] = inp[0].detach()

            handle = lin.register_forward_hook(_hook)
        except Exception:
            pass
        with torch.no_grad():
            out = model(x)
            if isinstance(out, list):
                out = out[1]
        if handle is not None:
            handle.remove()
        h = cap.get("h")
        if h is not None and h.dim() > 2:
            h = h.reshape(h.shape[0], -1)
        _SHARED.update(x=x, logits=out, h=h, enc=None)
    if need_enc and _SHARED["enc"] is None:
        with torch.no_grad():
            e = model.get_encoding(x)
            if e.dim() > 2:
                e = e.reshape(e.shape[0], -1)
        _SHARED["enc"] = e
    return _SHARED


def cross_entropy(model, image, num_features=1):
    s = _shared(model, image)
    return model.criterion(s["logits"], torch.ones_like(s["logits"]))


def _final_linear_weight(model):
    last = None
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            last = m
    if last is None:
        raise ValueError("GradNorm: model has no nn.Linear or nn.Conv2d layer")
    return last.weight


def _grad_magnitude_loop(model, x):
    """Original per-sample autograd GradNorm (fallback for non-Linear heads)."""
    final_weight = _final_linear_weight(model)
    scores = torch.empty(x.shape[0], device=x.device)
    with torch.enable_grad():
        for i in range(x.shape[0]):
            xi = x[i:i+1].detach()
            output = model(xi)
            if isinstance(output, list):
                output = output[1]
            log_probs = F.log_softmax(output, dim=1)
            ce = -log_probs.mean()
            model.zero_grad(set_to_none=True)
            grads = torch.autograd.grad(ce, final_weight, retain_graph=False)[0]
            scores[i] = grads.abs().sum()
    return scores


def grad_magnitude(model, x, num_features=1):
    """GradNorm OOD score (Huang et al. 2021), L1 gradient norm to a uniform target.

    Closed form for a linear classifier head: the gradient of the uniform-target
    cross-entropy w.r.t. the head weights is (softmax - 1/C) outer h, whose L1 norm
    factorizes as ||softmax - 1/C||_1 * ||h||_1, where h is the TRUE head input
    (captured via a forward hook, so this is exact for both ResNet and ViT — for
    ViT, get_encoding() is NOT the head input). Verified to 1e-7 vs the per-sample
    autograd loop. Falls back to the loop for models with no final nn.Linear.
    """
    s = _shared(model, x)
    if s["h"] is None:
        return _grad_magnitude_loop(model, x)
    logits, h = s["logits"], s["h"]
    C = logits.shape[1]
    sm = F.softmax(logits, dim=1)
    return (sm - 1.0 / C).abs().sum(1) * h.abs().sum(1)


def typicality(model, img, num_features=1):
    return -model.estimate_log_likelihood(img)


_KNN_REF = {"ref": None, "t": None}


def knn(model, img, train_test_norms):
    """Min L2 distance to the InD-train reference, vectorized (cdist) with the
    reference tensor cached on-device (was: per-sample Python loop + a full
    reference re-upload every batch). Exact match to the loop (1e-7). Cache keyed
    by object identity (`is`), not id(), to avoid id-reuse across cells."""
    global _KNN_REF
    z = _shared(model, img, need_enc=True)["enc"]
    if (_KNN_REF["ref"] is not train_test_norms or _KNN_REF["t"] is None
            or _KNN_REF["t"].device != z.device):
        R = torch.as_tensor(np.asarray(train_test_norms), dtype=torch.float32, device=z.device)
        R = R.view(-1, R.shape[-1])
        _KNN_REF.update(ref=train_test_norms, t=R)
    return torch.cdist(z.float(), _KNN_REF["t"]).min(dim=1).values


def energy(model, img, num_features=1):
    s = _shared(model, img)
    energy = torch.logsumexp(s["logits"], dim=1)
    while len(energy.shape) > 1:
        energy = torch.logsumexp(energy, dim=-1)
    return energy


def msp(model, img, num_features=1):
    s = _shared(model, img)
    feat = torch.max(F.softmax(s["logits"], dim=1), dim=1)[0]
    while len(feat.shape) != 1:
        feat = torch.max(feat, dim=-1)[0]
    return feat


# ---------------------------------------------------------------------------
# Latent-space / hybrid post-hoc detectors added for the revision (YLV1 W8).
# All are fit ONCE on the InD-train reference encodings (already capped by
# FeatureSD to keep memory/time bounded) and cached across batches.
# Detector orientation (higher => more OOD) is normalized by the OODDetector
# calibration downstream, so absolute sign is not critical.
# ---------------------------------------------------------------------------
_FIT = {}
_FIT_LAST_REF = None


def _final_linear(model):
    """Return the model's final nn.Linear (classifier head)."""
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    if last is None:
        raise ValueError("no nn.Linear head found for latent-space detector")
    return last


def _ref_matrix(train_enc):
    a = np.asarray(train_enc)
    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    return torch.as_tensor(a, dtype=torch.float32)  # CPU; fitting is one-time


def _ensure_fit(train_enc, model, kind, fitfn):
    """Fit `kind` on the reference encodings once per (dataset,model,mode) cell.

    Invalidated by OBJECT IDENTITY (`is`), not id() — FeatureSD holds one
    train_test_encodings array per cell, so identity is stable within a cell and
    changes across cells; id() is unsafe because a freed array's id can be reused
    (which silently reused a stale fit and crashed on the next architecture)."""
    global _FIT, _FIT_LAST_REF
    if train_enc is not _FIT_LAST_REF:
        _FIT = {}
        _FIT_LAST_REF = train_enc
    if kind not in _FIT:
        _FIT[kind] = fitfn(train_enc, model)
    return _FIT[kind]


def _encode(model, img):
    # Route through the shared per-batch forward cache (get_encoding computed once).
    return _shared(model, img, need_enc=True)["enc"]


def _to_dev(f, dev, keys):
    for k in keys:
        if hasattr(f[k], "device") and f[k].device != dev:
            f[k] = f[k].to(dev)


def _fit_maha(train_enc, model):
    Z = _ref_matrix(train_enc)
    mu = Z.mean(0)
    Zc = Z - mu
    cov = (Zc.T @ Zc) / (Z.shape[0] - 1)
    cov += 1e-4 * torch.eye(cov.shape[0])
    prec = torch.linalg.inv(cov)
    return {"mu": mu, "prec": prec}  # CPU; moved to input device on first use


def mahalanobis(model, img, train_test_encodings):
    """Marginal (class-agnostic) Mahalanobis distance in penultimate space.

    Class-conditional Mahalanobis needs train labels, which the collection
    pipeline does not expose to detectors; the marginal variant is a valid
    latent-space distance detector and needs only the reference encodings.
    """
    f = _ensure_fit(train_test_encodings, model, "maha", _fit_maha)
    z = _encode(model, img)
    _to_dev(f, z.device, ["mu", "prec"])
    zc = z - f["mu"]
    d = torch.einsum("bi,ij,bj->b", zc, f["prec"], zc)  # squared Mahalanobis
    return d


def _fit_react(train_enc, model):
    a = np.asarray(train_enc)
    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    if a.shape[0] > 20000:
        rng = np.random.default_rng(0)
        a = a[rng.choice(a.shape[0], 20000, replace=False)]
    c = float(np.percentile(a, 90))  # 90th-percentile activation clip (ReAct)
    return {"c": c}


def react(model, img, train_test_encodings):
    """ReAct (Sun et al. 2021): clip penultimate activations, then energy."""
    f = _ensure_fit(train_test_encodings, model, "react", _fit_react)
    lin = _final_linear(model)
    z = torch.clamp(_encode(model, img), max=f["c"])
    logits = F.linear(z, lin.weight, lin.bias)
    return torch.logsumexp(logits, dim=1)


def _fit_vim(train_enc, model):
    Z = _ref_matrix(train_enc)
    mu = Z.mean(0)
    Zc = Z - mu
    cov = (Zc.T @ Zc) / (Z.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    D = min(512, Z.shape[1] - 1)
    principal = evecs[:, -D:]  # top-D principal subspace
    proj = Zc @ principal
    res_norm = (Zc - proj @ principal.T).norm(dim=1)
    lin = _final_linear(model)
    w = lin.weight.detach().cpu()
    b = None if lin.bias is None else lin.bias.detach().cpu()
    logits = F.linear(Z, w, b)
    energy = torch.logsumexp(logits, dim=1)
    alpha = float(energy.mean() / (res_norm.mean() + 1e-8))
    return {"mu": mu, "principal": principal, "alpha": alpha}


def vim(model, img, train_test_encodings):
    """ViM (Wang et al. 2022): virtual-logit from feature residual vs. energy."""
    f = _ensure_fit(train_test_encodings, model, "vim", _fit_vim)
    lin = _final_linear(model)
    z = _encode(model, img)
    _to_dev(f, z.device, ["mu", "principal"])
    zc = z - f["mu"]
    proj = zc @ f["principal"]
    res_norm = (zc - proj @ f["principal"].T).norm(dim=1)
    vlogit = f["alpha"] * res_norm
    energy = torch.logsumexp(F.linear(z, lin.weight, lin.bias), dim=1)
    return vlogit - energy  # higher => more OOD (large residual, low energy)


# ---------------------------------------------------------------------------
# Classic anomaly detectors (unifying-view experiment). Each is fit on the InD
# reference encodings and scores test encodings; higher = more anomalous. These
# span paradigms distinct from the OOD-detector family: tree-isolation, local
# density (LOF), boundary (one-class SVM), and linear-subspace reconstruction.
# ---------------------------------------------------------------------------
def _ref_np(train_enc, cap):
    a = np.asarray(train_enc)
    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    if a.shape[0] > cap:
        a = a[np.random.default_rng(0).choice(a.shape[0], cap, replace=False)]
    return a.astype(np.float32)


def _fit_ad(train_enc, model, kind):
    if kind == "iforest":
        return IsolationForest(n_estimators=100, random_state=0).fit(_ref_np(train_enc, 15000))
    if kind == "lof":
        return LocalOutlierFactor(n_neighbors=20, novelty=True).fit(_ref_np(train_enc, 10000))
    if kind == "ocsvm":
        return OneClassSVM(nu=0.1, gamma="scale").fit(_ref_np(train_enc, 3000))
    if kind == "pca":
        a = _ref_np(train_enc, 15000); mu = a.mean(0)
        _, _, Vt = np.linalg.svd(a - mu, full_matrices=False)
        return {"mu": mu, "V": Vt[: min(64, Vt.shape[0])]}
    raise ValueError(kind)


def _ad_score(model, img, train_test_encodings, kind):
    f = _ensure_fit(train_test_encodings, model, "ad_" + kind, lambda te, m: _fit_ad(te, m, kind))
    z = _encode(model, img).detach().cpu().numpy().astype(np.float32)
    if kind == "pca":
        zc = z - f["mu"]; recon = (zc @ f["V"].T) @ f["V"]
        s = np.linalg.norm(zc - recon, axis=1)
    elif kind == "ocsvm":
        s = -f.decision_function(z)
    else:  # iforest, lof
        s = -f.score_samples(z)
    return torch.as_tensor(np.asarray(s, dtype=np.float32), device=img.device)


def isolation_forest(model, img, train_test_encodings):
    return _ad_score(model, img, train_test_encodings, "iforest")


def lof(model, img, train_test_encodings):
    return _ad_score(model, img, train_test_encodings, "lof")


def ocsvm(model, img, train_test_encodings):
    return _ad_score(model, img, train_test_encodings, "ocsvm")


def pca_recon(model, img, train_test_encodings):
    return _ad_score(model, img, train_test_encodings, "pca")
