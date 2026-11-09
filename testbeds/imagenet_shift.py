from testbeds.base import *
from datasets.imagenet_shift_datasets import build_imagenet_shift_dataset
from classifier.resnetclassifier import ResNetClassifier

# Base ImageNet val (ILSVRC2012), 1000 wnid subdirs. Must exist on the target machine.
IMAGENET_VAL = os.environ.get("IMAGENET_VAL", "../../Datasets/imagenet/val")
IMAGENET_SHIFT_ROOT = "../../Datasets/imagenet_shift"
VARIANT = os.environ.get("IMAGENET_VARIANT", "imagenet-v2")   # imagenet-v2 | imagenet-r | imagenet-a


class ImageNetShiftTestBed(BaseTestBed):
    """ImageNet with a pretrained backbone (NO training). Organic OOD = a shift variant.

    Uses the torchvision-pretrained 1000-class head so accuracy is meaningful without
    fine-tuning. Reuses the repo feature interface (get_encoding / latent_dim) via
    ResNetClassifier, then restores the pretrained fc weights.

    TODO (verify on-machine):
      * ViT backbone: mirror this with ViTClassifier + its pretrained head.
      * imagenet-r / imagenet-a cover 200 classes: mask logits to self.subset_idx
        before argmax in accuracy, else accuracy is understated. imagenet-v2 needs
        no masking, so it is the recommended first variant.
    """

    def __init__(self, model="resnet", mode="normal", sampler="RandomSampler", batch_size=32, pretrained=True):
        super().__init__(mode=mode, model=model, sampler=sampler, batch_size=batch_size)
        assert model == "resnet", "ViT path not wired yet (see class docstring TODO)."
        self.trans = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
        ])
        (_, self.ind_val, self.ind_test, self.ood_val, self.ood_test, self.subset_idx) = \
            build_imagenet_shift_dataset(IMAGENET_SHIFT_ROOT, IMAGENET_VAL, self.trans, variant=VARIANT)
        self.ind_train = self.ind_val               # no training; reference set = ind_val
        self.num_classes = 1000

        # pretrained backbone + restore pretrained 1000-class head
        import torchvision.models as tvm
        self.classifier = ResNetClassifier(1000, 101).to("cuda").eval()
        pre = tvm.resnet101(weights=tvm.ResNet101_Weights.IMAGENET1K_V2)
        self.classifier.model.fc.load_state_dict(pre.fc.state_dict())
        self.classifier = self.classifier.to("cuda").eval()
        self.glow = None
        self.mode = mode

    def get_ood_dict(self):
        return {f"{VARIANT} Val": self.dl(self.ood_val),
                f"{VARIANT} Test": self.dl(self.ood_test)}
