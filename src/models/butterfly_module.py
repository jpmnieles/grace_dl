from typing import Any, Dict, Tuple
import os

import numpy as np
import pandas as pd

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, MeanSquaredError
from torch.nn.functional import mse_loss, huber_loss

from src.models.components.reconlatent_loss import ReconLatentLoss


MAX_METER_VALUE = 1.5
MAX_DEG_VALUE = 44


class GraceLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        compile: bool,
        epsilon: float,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)

        self.net = net

        # loss function
        self.epsilon = epsilon
        self.criterion = ReconLatentLoss(self.epsilon)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking rmse
        self.train_rmse1 =  MeanSquaredError(squared=False)
        self.train_rmse2 =  MeanSquaredError(squared=False)
        self.train_latent_rmse =  MeanSquaredError(squared=False)
        self.val_rmse1 =  MeanSquaredError(squared=False)
        self.val_rmse2 =  MeanSquaredError(squared=False)
        self.val_latent_rmse =  MeanSquaredError(squared=False)
        self.test_rmse1 = MeanSquaredError(squared=False)
        self.test_rmse2 = MeanSquaredError(squared=False)
        self.test_latent_rmse = MeanSquaredError(squared=False)
        
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.val_rmse1_best = MinMetric()
        self.val_rmse2_best = MinMetric()
        self.val_latent_rmse_best = MinMetric()

        # test placeholder
        self._test_inputs = []
        self._test_preds1 = []
        self._test_preds2 = []
        self._test_latent1 = []
        self._test_latent2 = []
        self._test_targets1 = []
        self._test_targets2 = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_loss.reset()
        self.train_rmse1.reset()
        self.train_rmse2.reset()
        self.train_latent_rmse.reset()
        self.val_loss.reset()
        self.val_rmse1.reset()
        self.val_rmse2.reset()
        self.val_latent_rmse.reset()
        self.val_loss_best.reset()
        self.val_rmse1_best.reset()
        self.val_rmse2_best.reset()
        self.val_latent_rmse_best.reset()
        self.test_rmse1.reset()
        self.test_rmse2.reset()
        self.test_latent_rmse.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        output1, output2, latent1, latent2 = self.forward(x)
        target1 = y[:,:12]
        target2 = y[:,12:]
        loss = self.criterion(output1, output2, latent1, latent2, target1, target2)
        return loss, output1, output2, target1, target2, latent1, latent2

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, output1, output2, target1, target2, latent1, latent2 = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # RMSE calculation
        self.train_rmse1(output1, target1)
        self.train_rmse2(output2, target2)
        self.train_latent_rmse(latent1, latent2)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.log("train/rmse1", self.train_rmse1.compute(), sync_dist=True, prog_bar=True)
        self.log("train/rmse2", self.train_rmse2.compute(), sync_dist=True, prog_bar=True)
        self.log("train/latent_rmse", self.train_latent_rmse.compute(), sync_dist=True, prog_bar=True)
        self.train_rmse1.reset()
        self.train_rmse2.reset()
        self.train_latent_rmse.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, output1, output2, target1, target2, latent1, latent2 = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # RMSE calculation
        self.val_rmse1(output1, target1)
        self.val_rmse2(output2, target2)
        self.val_latent_rmse(latent1, latent2)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

        rmse1 = self.val_rmse1.compute()
        self.val_rmse1_best(rmse1)
        self.log("val/rmse1", rmse1, sync_dist=True, prog_bar=True)
        self.log("val/rmse1_best", self.val_rmse1_best.compute(), sync_dist=True, prog_bar=True)
        self.val_rmse1.reset()

        rmse2 = self.val_rmse2.compute()
        self.val_rmse2_best(rmse2)
        self.log("val/rmse2", rmse2, sync_dist=True, prog_bar=True)
        self.log("val/rmse2_best", self.val_rmse2_best.compute(), sync_dist=True, prog_bar=True)
        self.val_rmse2.reset()

        latent_rmse = self.val_latent_rmse.compute()
        self.val_latent_rmse_best(latent_rmse)
        self.log("val/latent_rmse", latent_rmse, sync_dist=True, prog_bar=True)
        self.log("val/latent_rmse_best", self.val_latent_rmse_best.compute(), sync_dist=True, prog_bar=True)
        self.val_latent_rmse.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, output1, output2, target1, target2, latent1, latent2 = self.model_step(batch)

        # list append
        x, _ = batch
        self._test_inputs.append(x.detach().cpu().numpy())
        self._test_preds1.append(output1.detach().cpu().numpy())
        self._test_targets1.append(target1.detach().cpu().numpy())
        self._test_preds2.append(output2.detach().cpu().numpy())
        self._test_targets2.append(target2.detach().cpu().numpy())
        self._test_latent1.append(latent1.detach().cpu().numpy())
        self._test_latent2.append(latent2.detach().cpu().numpy())

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        # RMSE calculation
        self.test_rmse1(output1, target1)
        self.test_rmse2(output2, target2)
        self.test_latent_rmse(latent1, latent2)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("test/rmse1", self.test_rmse1, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("test/rmse2", self.test_rmse2, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("test/latent_rmse", self.test_latent_rmse, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log("test/rmse1", self.test_rmse1.compute(), prog_bar=True)
        self.log("test/rmse2", self.test_rmse2.compute(), prog_bar=True)
        self.log("test/latent_rmse", self.test_latent_rmse.compute(), prog_bar=True)
        
        # Saving the CSV for analysis
        inputs_arr = np.concatenate(self._test_inputs, axis=0)
        preds1_arr = np.concatenate(self._test_preds1, axis=0)
        targets1_arr = np.concatenate(self._test_targets1, axis=0)
        preds2_arr = np.concatenate(self._test_preds2, axis=0)
        targets2_arr = np.concatenate(self._test_targets2, axis=0)
        latent1_arr = np.concatenate(self._test_latent1, axis=0)
        latent2_arr = np.concatenate(self._test_latent2, axis=0)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = GraceLitModule(None, None, None, 0.5)
