from testbeds.base import *
from datasets.cifar_datasets import build_cifar10_dataset, cifar10c_folds
from classifier.resnetclassifier import ResNetClassifier
from classifier.vit import ViTClassifier

CIFAR_ROOT = "../../Datasets"          # contains cifar/ (see scripts/download_uae_datasets.sh)
INCLUDE_CIFAR10C = True                # add CIFAR-10-C corruptions as extra organic folds


class CIFAR10TestBed(BaseTestBed):
    """CIFAR-10 base; organic OOD = CIFAR-10.1 (+ optional CIFAR-10-C). No Glow."""

    def __init__(self, model="resnet", mode="normal", sampler="RandomSampler", batch_size=32, pretrained=True):
        super().__init__(mode=mode, model=model, sampler=sampler, batch_size=batch_size)
        self.trans = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
        ])
        self.ind_train, self.ind_val, self.ind_test, self.ood_val, self.ood_test = \
            build_cifar10_dataset(CIFAR_ROOT, self.trans, self.trans)
        self.num_classes = num_classes = self.ind_train.num_classes
        prefix = "classifier_logs"
        if model == "resnet":
            self.classifier = ResNetClassifier.load_from_checkpoint(
                f"{prefix}/{model}/CIFAR10/checkpoints/best.ckpt",
                num_classes=num_classes, resnet_version=101).to("cuda").eval()
        else:
            self.classifier = ViTClassifier.load_from_checkpoint(
                f"{prefix}/{model}/CIFAR10/checkpoints/best.ckpt",
                num_classes=num_classes).to("cuda").eval()
        self.glow = None
        self.mode = mode

    def get_ood_dict(self):
        d = {"CIFAR-10.1 Val": self.dl(self.ood_val),
             "CIFAR-10.1 Test": self.dl(self.ood_test)}
        if INCLUDE_CIFAR10C:
            for name, ds in cifar10c_folds(CIFAR_ROOT, self.trans).items():
                d[name] = self.dl(ds)
        return d
