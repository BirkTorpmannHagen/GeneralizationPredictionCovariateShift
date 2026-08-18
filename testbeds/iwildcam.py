from testbeds.base import *
from datasets.wilds_datasets import build_iwildcam_dataset
from classifier.resnetclassifier import ResNetClassifier
from classifier.vit import ViTClassifier

WILDS_ROOT = "../../Datasets/wilds"


class IWildCamTestBed(BaseTestBed):
    """WILDS iWildCam (domain = camera-trap location). Typicality/Glow detector is skipped."""

    def __init__(self, model="resnet", mode="normal", sampler="RandomSampler", batch_size=16, pretrained=True):
        super().__init__(mode=mode, model=model, sampler=sampler, batch_size=batch_size)
        self.trans = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
        ])
        # Full evaluation folds (no subsampling). The kNN/Maha/ViM reference is
        # capped inside FeatureSD (REFERENCE_CAP), not here.
        self.ind_train, self.ind_val, self.ind_test, self.ood_val, self.ood_test = \
            build_iwildcam_dataset(WILDS_ROOT, self.trans, self.trans)
        self.num_classes = num_classes = self.ind_train.num_classes
        prefix = "classifier_logs"
        if model == "resnet":
            self.classifier = ResNetClassifier.load_from_checkpoint(
                f"{prefix}/{model}/IWildCam/checkpoints/best.ckpt",
                num_classes=num_classes, resnet_version=101).to("cuda").eval()
        else:
            self.classifier = ViTClassifier.load_from_checkpoint(
                f"{prefix}/{model}/IWildCam/checkpoints/best.ckpt",
                num_classes=num_classes).to("cuda").eval()
        self.glow = None
        self.mode = mode

    def get_ood_dict(self):
        return {"OoD Val": self.dl(self.ood_val),
                "OoD Test": self.dl(self.ood_test)}
