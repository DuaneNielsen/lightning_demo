from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Tuple, Union
from argparse import ArgumentParser, Namespace
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.utilities.argparse import from_argparse_args
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.models.vision.unet import UNet
import wandb
import numpy as np
from cityscape_labels import labels

class_labels =  {label.id: label.name for label in labels}

class WandbImagePredCallback(pl.Callback):
    """Logs the input images and output predictions of a module.

    Predictions and labels are logged as class indices."""

    def __init__(self, num_samples=32):
        super().__init__()
        self.num_samples = num_samples

    def masked_image(self, val_img, preds, labels):
        preds = preds.cpu().numpy()
        return wandb.Image(val_img, masks={
            "predictions": {
                "mask_data": preds,
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": labels,
                "class_labels": class_labels
            }
        })

    def on_validation_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            val_loader = trainer.datamodule.val_dataloader()
            val_data = []
            for i in range(self.num_samples):
                val_data += [val_loader.dataset[i]]
            val_data = list(zip(*val_data))
            val_imgs = torch.stack(val_data[0]).to(device=pl_module.device)
            val_labels = np.stack(val_data[1])
            logits = pl_module(val_imgs)
            preds = torch.argmax(logits, dim=1)
            trainer.logger.experiment.log({"img_with_masks": self.masked_image(val_imgs[0], preds[0], val_labels[0])
                                              , "global_step": trainer.global_step})


class SemSegment(LightningModule):
    def __init__(
        self,
        lr: float = 0.01,
        num_classes: int = 19,
        num_layers: int = 5,
        features_start: int = 64,
        bilinear: bool = False,
    ):
        """Basic model for semantic segmentation. Uses UNet architecture by default.
        The default parameters in this model are for the KITTI dataset. Note, if you'd like to use this model as is,
        you will first need to download the KITTI dataset yourself. You can download the dataset `here.
        <http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015>`_
        Implemented by:
            - `Annika Brundyn <https://github.com/annikabrundyn>`_
        Args:
            num_layers: number of layers in each side of U-net (default 5)
            features_start: number of features in first layer (default 64)
            bilinear: whether to use bilinear interpolation (True) or transposed convolutions (default) for upsampling.
            lr: learning (default 0.01)
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear
        self.lr = lr

        self.net = UNet(
            num_classes=num_classes,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        log_dict = {"train_loss": loss_val}
        return {"loss": loss_val, "log": log_dict, "progress_bar": log_dict}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            img, mask = batch
            img = img.float()
            mask = mask.long()
            out = self(img)
            loss_val = F.cross_entropy(out, mask, ignore_index=250)
            return {"val_loss": loss_val}

    def validation_epoch_end(self, outputs):
        with torch.no_grad():
            loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
            log_dict = {"val_loss": loss_val}
            return {"log": log_dict, "val_loss": log_dict["val_loss"], "progress_bar": log_dict}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
        parser.add_argument("--num_layers", type=int, default=5, help="number of layers on u-net")
        parser.add_argument("--features_start", type=float, default=64, help="number of features in first layer")
        parser.add_argument(
            "--bilinear", action="store_true", default=False, help="whether to use bilinear interpolation or transposed"
        )

        return parser

    @classmethod
    def from_argparse_args(cls: Any, args: Union[Namespace, ArgumentParser], **kwargs) -> Any:
        return from_argparse_args(cls, args, **kwargs)


def cli_main():
    from pl_bolts.datamodules import KittiDataModule

    seed_everything(1234)
    wandb_logger =  WandbLogger(project='lightning_demo_unet_kitti')

    # get arguments
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = SemSegment.add_model_specific_args(parser)
    parser = KittiDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    # data
    dm = KittiDataModule(args.data_dir).from_argparse_args(args)

    # model
    model = SemSegment().from_argparse_args(args)

    # train
    trainer = Trainer().from_argparse_args(args,
                                           callbacks=[WandbImagePredCallback(num_samples=2)],
                                           logger=wandb_logger)
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    cli_main()