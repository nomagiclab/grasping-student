import random

import clearml
import pytorch_lightning as pl
import torch
import torch.utils.data
from clearml import Task
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import data
from torch import optim
from torch.nn import BCELoss
import dataset
from dataset.pointwise import ExtendedPickingDataset, PickingDataset
from model.metrics import ImitationLearningPointwiseMetrics
from model.repository import SegmentationModelRepository
from model.rotations import AffordanceRotationsModule
import segmentation_models_pytorch as smp


class ImitationLearning(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-5, max_epochs=10):
        super().__init__()
        self.save_hyperparameters()
        model = smp.Unet(
            encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,
            # encoder_depth=4, # model output channels (number of classes in your dataset)
        )
        self.backbone = model  #SegmentationModelRepository.fcn_resnet_50_rgbd(nchannels=1)
        self.backbone = AffordanceRotationsModule(
            self.backbone,
            num_rotations=64,
            padding_noise=0.01
        )
        self.device_ = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def get_angles(self, angle_index):
        return [angle_index[0]] + random.sample([x for x in range(64) if abs(x - angle_index[0]) > 10], 1)

    def pointed_bce_loss(self, batch, mode="train"):
        idxs = batch["grasping_index"]
        h = batch["heightmap"][:, 3] if batch["heightmap"].shape[1] > 1 else batch["heightmap"][:, 0]
        angles = self.get_angles(idxs["angle_index"])
        affordances = self.backbone(
            h.to(self.device_),
            inference=(mode == "val"),
            idxs=angles,
        )

        idxs["angle_index"] = [0]
        affordance = torch.stack([
            affordances[i, :, x, y, a]
            for i, x, y, a in zip(range(len(idxs["row"])), list(idxs["row"]), list(idxs["col"]), list(idxs["angle_index"]))
        ])

        successful = torch.unsqueeze(torch.tensor(batch["successful"], dtype=torch.float, device=affordance.device), dim=1)

        loss = BCELoss()(affordance, successful)

        metrics = ImitationLearningPointwiseMetrics(affordances, idxs, batch["successful"], losses=loss)
        for k, v in metrics.summary().items():
            self.log(f"{mode}_{k}", v, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.pointed_bce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.pointed_bce_loss(batch, mode="val")


if __name__ == "__main__":
    task = Task.init(project_name="krzywicki", task_name=f"imitation")
    root_dir = clearml.Dataset.get(dataset_name="all-grasps", dataset_project="nomagic").get_local_copy()
    picking_dataset = PickingDataset(root_dir)

    train_len = int(0.8 * (len(picking_dataset)))
    train_set, val_set = torch.utils.data.random_split(picking_dataset, [train_len, len(picking_dataset) - train_len])

    def train(batch_size=1, max_epochs=10, **kwargs):
        trainer = pl.Trainer(
            default_root_dir="./",
            # logger=TensorBoardLogger('../'),
            gpus=1 if torch.cuda.is_available() else 0,
            max_epochs=max_epochs,
            callbacks=[
                ModelCheckpoint(save_weights_only=True, every_n_epochs=1),
                LearningRateMonitor("epoch"),
            ],
            # progress_bar_refresh_rate=1,
            log_every_n_steps=1,
        )
        train_loader = data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        val_loader = data.DataLoader(
            val_set,
            batch_size=batch_size,
            drop_last=True,
            pin_memory=True,
        )

        model = ImitationLearning(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = ImitationLearning.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        return model

    model = train()
