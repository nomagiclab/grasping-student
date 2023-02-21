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

from dataset.pointwise import ExtendedPickingDataset, PickingDataset, AugmentedPickingDataset
from model.affordance_learning import AffordanceLearning
from model.imitation_learning import ImitationLearning
from model.metrics import ImitationLearningPointwiseMetrics, topn_masks
from model.rotations import AffordanceRotationsModule
import segmentation_models_pytorch as smp

fraction = 1.0
topn_range = 2

class StudentLearning(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-5, max_epochs=10):
        super().__init__()
        self.save_hyperparameters()
        self.device_ = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        model = smp.FPN(
            encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,
            # encoder_depth=4, # model output channels (number of classes in your dataset)
        )
        self.backbone = AffordanceRotationsModule(
            model,
            num_rotations=64,
            padding_noise=0.01
        )
        self.affordance_model = AffordanceLearning.load_from_checkpoint("../artifacts/affordance_model.ckpt", strict=False).to(self.device_)
        self.teacher_model = ImitationLearning.load_from_checkpoint(f"../artifacts/teacher-{fraction}.ckpt", strict=False).to(self.device_)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def get_angles(self, angle_index):
        return torch.tensor([random.sample([x for x in range(64) if abs(x - angle) > 10], 1)[0] for angle in angle_index])

    def get_idxs(self, h, nth_top=1):
        h = h[:, 3:4] if h.shape[1] > 1 else h[:, 0:1]
        h = h.to(self.device_)
        angles_sample = list(range(64))
        angles_sample = random.sample(list(range(64)), 16)

        with torch.no_grad():
            teacher_outputs = self.teacher_model.backbone(
                h,
                inference=True,
                idxs=[x * torch.ones((h.shape[0])).to(self.device_) for x in angles_sample],
                softmax_at_the_end=True,
            )

            def unravel_index(index, shape):
                out = []
                for dim in reversed(shape):
                    out.append(index % dim)
                    index = index // dim
                return tuple(reversed(out))

            topn = topn_masks(teacher_outputs)
            masks = (topn[nth_top - 1] * 1.) - ((topn[nth_top - 2] * 1.) if nth_top - 2 > 0 else 0)
            results = [unravel_index(torch.argmax(mask), mask.shape)[1:] for mask in masks]
            rows, cols, angles = zip(*results)

            return {
                "row": torch.tensor(rows, device=self.device_),
                "col": torch.tensor(cols, device=self.device_),
                "angle_index": torch.tensor([angles_sample[a] for a in angles], device=self.device_),
            }

    def pointed_bce_loss(self, batch, mode="train"):
        h = batch["heightmap"][:, 3:4] if batch["heightmap"].shape[1] > 1 else batch["heightmap"][:, 0:1]
        h = h.to(self.device_)
        nth_top = random.randint(1, topn_range)
        idxs = self.get_idxs(h, nth_top=nth_top)
        # else:
        #     idxs = batch["grasping_index"]
        affordances = self.backbone(
            h,
            inference=(mode == "val"),
            idxs=[idxs["angle_index"], self.get_angles(idxs["angle_index"]).to(self.device_)],
            softmax_at_the_end=True,
        )

        with torch.no_grad():
            ground_truth = self.affordance_model.backbone(
                h,
                inference=True,
                idxs=[idxs["angle_index"]],
                softmax_at_the_end=False,
            )
            ground_truth = torch.sigmoid(ground_truth) > 0.5

        affordance = torch.stack([
            affordances[i, :, x.item(), y.item(), 0]
            for i, x, y, a in zip(range(len(idxs["row"])), list(idxs["row"]), list(idxs["col"]), list(idxs["angle_index"]))
        ])

        successful = torch.unsqueeze(torch.tensor(torch.ones_like(batch["successful"]), dtype=torch.float, device=affordance.device), dim=1)

        loss = BCELoss()(affordance, successful)

        with torch.no_grad():
            metrics = ImitationLearningPointwiseMetrics(affordances, idxs, batch["successful"], losses=loss, ground_truth=ground_truth)
            for k, v in metrics.summary().items():
                self.log(f"{mode}_{k}", v, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.pointed_bce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.pointed_bce_loss(batch, mode="val")


if __name__ == "__main__":
    task = Task.init(project_name="krzywicki", task_name=f"final-student-learning-{fraction}-debug")
    picking_dataset = PickingDataset(clearml.Dataset.get(dataset_name="all-grasps", dataset_project="nomagic").get_local_copy())
    picking_dataset = AugmentedPickingDataset(picking_dataset, overwrite_num_rotations=64)

    train_len = int(0.9 * (len(picking_dataset)))
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

        model = StudentLearning(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)

    train(batch_size=8, lr=1e-3, max_epochs=30 * topn_range)
