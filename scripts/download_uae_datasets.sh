#!/usr/bin/env bash
# Download the standard UAE / OOD-performance-prediction benchmark datasets we are
# missing (see task P1 / UAE_DATA_COLLECTION.md). Idempotent: skips anything already
# present. Set DATA_ROOT to where you want them (default ../Datasets relative to repo).
#
#   bash scripts/download_uae_datasets.sh cifar        # just the CIFAR family
#   bash scripts/download_uae_datasets.sh imagenet     # ImageNet shift variants
#   bash scripts/download_uae_datasets.sh wilds        # FMoW + RxRx1
#   bash scripts/download_uae_datasets.sh all          # everything below
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)/Datasets}"
WHICH="${1:-all}"
mkdir -p "$DATA_ROOT"
echo "[download] DATA_ROOT=$DATA_ROOT   target=$WHICH"

have() { [ -e "$1" ]; }
fetch() {  # fetch <url> <outfile>
  if have "$2"; then echo "  [skip] $2 exists"; return; fi
  echo "  [get ] $1 -> $2"
  curl -L --fail --retry 3 -o "$2" "$1"
}

# ---------------------------------------------------------------- CIFAR family
if [ "$WHICH" = "cifar" ] || [ "$WHICH" = "all" ]; then
  echo "== CIFAR-10 family =="
  CDIR="$DATA_ROOT/cifar"; mkdir -p "$CDIR"
  # base CIFAR-10 is auto-downloaded by torchvision at build time; nothing to do here.
  # CIFAR-10-C (synthetic corruption suite, 15 types x 5 severities as .npy)
  if ! have "$CDIR/CIFAR-10-C/labels.npy"; then
    fetch "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar" "$CDIR/CIFAR-10-C.tar"
    tar -xf "$CDIR/CIFAR-10-C.tar" -C "$CDIR" && rm -f "$CDIR/CIFAR-10-C.tar"
  else echo "  [skip] CIFAR-10-C present"; fi
  # CIFAR-10.1 (natural shift, v6 .npy: data + labels)
  mkdir -p "$CDIR/CIFAR-10.1"
  fetch "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_data.npy"   "$CDIR/CIFAR-10.1/cifar10.1_v6_data.npy"
  fetch "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_labels.npy" "$CDIR/CIFAR-10.1/cifar10.1_v6_labels.npy"
  # CINIC-10 (ImageFolder: train/valid/test x 10 classes) -- optional, larger
  if ! have "$CDIR/CINIC-10/test"; then
    fetch "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz" "$CDIR/CINIC-10.tar.gz"
    mkdir -p "$CDIR/CINIC-10"; tar -xzf "$CDIR/CINIC-10.tar.gz" -C "$CDIR/CINIC-10" && rm -f "$CDIR/CINIC-10.tar.gz"
  else echo "  [skip] CINIC-10 present"; fi
fi

# ---------------------------------------------------------------- ImageNet shift
if [ "$WHICH" = "imagenet" ] || [ "$WHICH" = "all" ]; then
  echo "== ImageNet shift variants =="
  IDIR="$DATA_ROOT/imagenet_shift"; mkdir -p "$IDIR"
  echo "  NOTE: base ImageNet val (ILSVRC2012) is NOT downloadable here; point"
  echo "        IMAGENET_VAL at your existing copy (ImageFolder of 1000 wnid dirs)."
  # ImageNet-R (200-class renditions)
  if ! have "$IDIR/imagenet-r"; then
    fetch "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar" "$IDIR/imagenet-r.tar"
    tar -xf "$IDIR/imagenet-r.tar" -C "$IDIR" && rm -f "$IDIR/imagenet-r.tar"
  else echo "  [skip] imagenet-r present"; fi
  # ImageNet-A (200-class natural adversarial)
  if ! have "$IDIR/imagenet-a"; then
    fetch "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar" "$IDIR/imagenet-a.tar"
    tar -xf "$IDIR/imagenet-a.tar" -C "$IDIR" && rm -f "$IDIR/imagenet-a.tar"
  else echo "  [skip] imagenet-a present"; fi
  # ImageNet-V2 (matched-frequency, 1000-class clean natural shift)
  if ! have "$IDIR/imagenetv2-matched-frequency-format-val"; then
    fetch "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz" "$IDIR/imagenetv2-mf.tar.gz"
    tar -xzf "$IDIR/imagenetv2-mf.tar.gz" -C "$IDIR" && rm -f "$IDIR/imagenetv2-mf.tar.gz"
  else echo "  [skip] imagenet-v2 present"; fi
  echo "  ImageNet-C is large (~75 GB across 5 tars). Uncomment in the script to pull:"
  echo "    zenodo record 2235448 (blur/digital/extra/noise/weather tars)"
fi

# ---------------------------------------------------------------- WILDS (FMoW, RxRx1)
if [ "$WHICH" = "wilds" ] || [ "$WHICH" = "all" ]; then
  echo "== WILDS FMoW + RxRx1 (via the wilds package) =="
  WROOT="$DATA_ROOT/wilds"; mkdir -p "$WROOT"
  python - "$WROOT" <<'PY'
import sys
from wilds import get_dataset
root = sys.argv[1]
for name in ["fmow", "rxrx1"]:
    print(f"  [wilds] get_dataset({name}) -> {root}")
    get_dataset(dataset=name, root_dir=root, download=True)
PY
fi

# ---------------------------------------------------------------- DomainNet (optional)
if [ "$WHICH" = "domainnet" ] || [ "$WHICH" = "all" ]; then
  echo "== DomainNet (6 domains) =="
  DDIR="$DATA_ROOT/domainnet"; mkdir -p "$DDIR"
  for dom in clipart infograph painting quickdraw real sketch; do
    fetch "http://csr.bu.edu/ftp/visda/2019/multi-source/${dom}.zip" "$DDIR/${dom}.zip"
    if [ ! -d "$DDIR/$dom" ]; then unzip -q "$DDIR/${dom}.zip" -d "$DDIR" && rm -f "$DDIR/${dom}.zip"; fi
  done
fi

echo "[download] done. Datasets under $DATA_ROOT"
