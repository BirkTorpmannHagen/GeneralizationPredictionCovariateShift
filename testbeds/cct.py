from testbeds.base import *

class CCTTestBed(BaseTestBed):
    def __init__(self, model="resnet", mode="severity", sampler="RandomSampler", batch_size=16, pretrained=True):
        super().__init__( mode=mode, model=model, sampler=sampler, batch_size=batch_size)
        self.trans = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(), ])
        self.ind_train, self.ind_val, self.ind_test, self.ood_val, self.ood_test= build_cct_dataset("../../Datasets/CCT", self.trans, self.trans)
        self.num_classes = num_classes = self.ind_train.num_classes
        prefix = "classifier_logs"
        if model=="resnet":
            self.classifier = ResNetClassifier.load_from_checkpoint(
                f"{prefix}/{model}/CCT/checkpoints/best.ckpt", num_classes=num_classes,
                resnet_version=101).to("cuda").eval()
        else:
            self.classifier = ViTClassifier.load_from_checkpoint(f"{prefix}/{model}/CCT/checkpoints/best.ckpt", num_classes=num_classes,)

        self.glow = GlowPL.load_from_checkpoint("glow_logs/CCT/checkpoints/epoch=199-step=166600.ckpt",in_channel=3, n_flow=32, n_block=4, affine=True, conv_lu=True,).cuda().eval()

        self.mode = mode

    def get_ood_dict(self):
        return {"OoD Val":self.dl(self.ood_val),
                "OoD Test":self.dl(self.ood_test)}
