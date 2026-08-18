from testbeds.base import *

class NICOTestBed(BaseTestBed):

    def __init__(self, model="resnet", mode="severity", sampler="RandomSampler", batch_size=16, pretrained=True):
        super().__init__( model=model, mode=mode, sampler=sampler, batch_size=batch_size)
        self.trans = transforms.Compose([
                                                 transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                                                 transforms.ToTensor(), ])
        self.num_classes = num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))

        self.num_classes = len(os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train/dim"))
        self.contexts = os.listdir("../../Datasets/NICO++/track_1/public_dg_0416/train")
        self.ind_train, self.ind_val, self.ind_test, self.oods = build_nico_dataset( "../../Datasets/NICO++", self.trans, self.trans, ind_context="dim")
        self.contexts.remove("dim")

        prefix = "classifier_logs"
        if model == "resnet":
            self.classifier = ResNetClassifier.load_from_checkpoint(
                f"{prefix}/{model}/NICO/checkpoints/best.ckpt", num_classes=num_classes,
                resnet_version=101).to("cuda").eval()
        else:
            self.classifier = ViTClassifier.load_from_checkpoint(f"{prefix}/{model}/NICO/checkpoints/best.ckpt",
                                                                 num_classes=num_classes, ).to("cuda").eval()
        self.glow = GlowPL.load_from_checkpoint("glow_logs/NICODataset/checkpoints/epoch=499-step=312500.ckpt",  in_channel=3, n_flow=32, n_block=4, conv_lu=True, affine=True).cuda().eval()
        self.mode=mode

    def get_ood_dict(self):
        return {dataset_name: self.dl(dataset) for dataset_name, dataset in self.oods.items()}
