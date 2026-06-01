from segmentor.segmentor import SegmentationModel
from pytorch_lightning import Trainer
from datasets.polyps import build_polyp_dataset
from torch.utils.data import DataLoader
import warnings
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import INPUT_SIZE

warnings.filterwarnings('ignore')

#

def train_segmentor():
    model_names = [ "deeplabv3plus"]
    for model_name in model_names:
        model = SegmentationModel(batch_size=16, model_name=model_name)
        logger = TensorBoardLogger(save_dir="segmentation_logs")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",           # Metric to monitor
            mode="min",                   # Minimize the metric for best model (use "max" for metrics where higher is better)
            save_top_k=1,                 # Save only the best model
            dirpath=f"segmentation_logs/checkpoints/{model_name}",  # Directory to save the checkpoint
            filename="best"    # Filename for the checkpoint
        )
        trainer = Trainer(accelerator="gpu", max_epochs=300,logger=logger, callbacks=[checkpoint_callback])

        ind, val, test, _, _, _ = build_polyp_dataset("../../Datasets/Polyps")
        train_loader = DataLoader(ind, batch_size=16, shuffle=True, num_workers=4)
        val_loader = DataLoader(val, batch_size=16, shuffle=True, num_workers=4)
        trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)

if __name__ == '__main__':
    train_segmentor()

