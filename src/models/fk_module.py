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
MAX_RVEC_VALUE = 1.5708


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
        self.criterion = mse_loss

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking rmse
        self.train_rmse =  MeanSquaredError(squared=False)
        self.val_rmse=  MeanSquaredError(squared=False)
        self.test_rmse = MeanSquaredError(squared=False)
        
        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        self.val_rmse_best = MinMetric()

        # test placeholder
        self._test_inputs = []
        self._test_preds = []
        self._test_latent = []
        self._test_targets = []

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
        self.train_rmse.reset()
        self.val_loss.reset()
        self.val_rmse.reset()
        self.val_loss_best.reset()
        self.val_rmse_best.reset()
        self.test_rmse.reset()

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
        preds, latents = self.forward(x)
        loss = self.criterion(preds, y)
        return loss, preds, y, latents

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, latents = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # RMSE calculation
        self.train_rmse(preds, targets)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.log("train/rmse1", self.train_rmse.compute(), sync_dist=True, prog_bar=True)
        self.train_rmse.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, latents = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # RMSE calculation
        self.val_rmse(preds, targets)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

        rmse1 = self.val_rmse.compute()
        self.val_rmse_best(rmse1)
        self.log("val/rmse1", rmse1, sync_dist=True, prog_bar=True)
        self.log("val/rmse1_best", self.val_rmse_best.compute(), sync_dist=True, prog_bar=True)
        self.val_rmse.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, latents = self.model_step(batch)

        # list append
        x, _ = batch
        self._test_inputs.append(x.detach().cpu().numpy())
        self._test_preds.append(preds.detach().cpu().numpy())
        self._test_targets.append(targets.detach().cpu().numpy())
        self._test_latent.append(latents.detach().cpu().numpy())

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        # RMSE calculation
        self.test_rmse(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("test/rmse1", self.test_rmse, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log("test/rmse1", self.test_rmse.compute(), prog_bar=True)
        
        # Saving the CSV for analysis
        inputs_arr = np.concatenate(self._test_inputs, axis=0)
        preds_arr = np.concatenate(self._test_preds, axis=0)
        targets_arr = np.concatenate(self._test_targets, axis=0)

        # Reorganizing DataFrame
        df = pd.DataFrame({
            'cmd_lnt_tplus1': MAX_DEG_VALUE*inputs_arr[:,0],
            'cmd_lnp_tplus1': MAX_DEG_VALUE*inputs_arr[:,1],
            'cmd_unt_tplus1': MAX_DEG_VALUE*inputs_arr[:,2],
            'cmd_et_tplus1': MAX_DEG_VALUE*inputs_arr[:,3],
            'cmd_lep_tplus1': MAX_DEG_VALUE*inputs_arr[:,4],
            'cmd_rep_tplus1': MAX_DEG_VALUE*inputs_arr[:,5],
            'cmd_lnt_t': MAX_DEG_VALUE*inputs_arr[:,6],
            'cmd_lnp_t': MAX_DEG_VALUE*inputs_arr[:,7],
            'cmd_unt_t': MAX_DEG_VALUE*inputs_arr[:,8],
            'cmd_et_t': MAX_DEG_VALUE*inputs_arr[:,9],
            'cmd_lep_t': MAX_DEG_VALUE*inputs_arr[:,10],
            'cmd_rep_t': MAX_DEG_VALUE*inputs_arr[:,11],
            
            'target_l_rvec_0': MAX_RVEC_VALUE*targets_arr[:,0],
            'target_l_rvec_1': MAX_RVEC_VALUE*targets_arr[:,1],
            'target_l_rvec_2': MAX_RVEC_VALUE*targets_arr[:,2],
            'target_l_tvec_0': MAX_METER_VALUE*targets_arr[:,3],
            'target_l_tvec_1': MAX_METER_VALUE*targets_arr[:,4],
            'target_l_tvec_2': MAX_METER_VALUE*targets_arr[:,5],
            'target_r_rvec_0': MAX_RVEC_VALUE*targets_arr[:,6],
            'target_r_rvec_1': MAX_RVEC_VALUE*targets_arr[:,7],
            'target_r_rvec_2': MAX_RVEC_VALUE*targets_arr[:,8],
            'target_r_tvec_0': MAX_METER_VALUE*targets_arr[:,9],
            'target_r_tvec_1': MAX_METER_VALUE*targets_arr[:,10],
            'target_r_tvec_2': MAX_METER_VALUE*targets_arr[:,11],
            
            'pred_l_rvec_0': MAX_RVEC_VALUE*preds_arr[:,0],
            'pred_l_rvec_1': MAX_RVEC_VALUE*preds_arr[:,1],
            'pred_l_rvec_2': MAX_RVEC_VALUE*preds_arr[:,2],
            'pred_l_tvec_0': MAX_METER_VALUE*preds_arr[:,3],
            'pred_l_tvec_1': MAX_METER_VALUE*preds_arr[:,4],
            'pred_l_tvec_2': MAX_METER_VALUE*preds_arr[:,5],
            'pred_r_rvec_0': MAX_RVEC_VALUE*preds_arr[:,6],
            'pred_r_rvec_1': MAX_RVEC_VALUE*preds_arr[:,7],
            'pred_r_rvec_2': MAX_RVEC_VALUE*preds_arr[:,8],
            'pred_r_tvec_0': MAX_METER_VALUE*preds_arr[:,9],
            'pred_r_tvec_1': MAX_METER_VALUE*preds_arr[:,10],
            'pred_r_tvec_2': MAX_METER_VALUE*preds_arr[:,11],
        })

        # Delta Analysis
        df['delta_l_rvec_0'] = df['target_l_rvec_0'].values - df['pred_l_rvec_0'].values
        df['delta_l_rvec_1'] = df['target_l_rvec_1'].values - df['pred_l_rvec_1'].values
        df['delta_l_rvec_2'] = df['target_l_rvec_2'].values - df['pred_l_rvec_2'].values
        df['delta_l_tvec_0'] = df['target_l_tvec_0'].values - df['pred_l_tvec_0'].values
        df['delta_l_tvec_1'] = df['target_l_tvec_1'].values - df['pred_l_tvec_1'].values
        df['delta_l_tvec_2'] = df['target_l_tvec_2'].values - df['pred_l_tvec_2'].values
        df['delta_r_rvec_0'] = df['target_r_rvec_0'].values - df['pred_r_rvec_0'].values
        df['delta_r_rvec_1'] = df['target_r_rvec_1'].values - df['pred_r_rvec_1'].values
        df['delta_r_rvec_2'] = df['target_r_rvec_2'].values - df['pred_r_rvec_2'].values
        df['delta_r_tvec_0'] = df['target_r_tvec_0'].values - df['pred_r_tvec_0'].values
        df['delta_r_tvec_1'] = df['target_r_tvec_1'].values - df['pred_r_tvec_1'].values
        df['delta_r_tvec_2'] = df['target_r_tvec_2'].values - df['pred_r_tvec_2'].values

        csv_path = os.path.join(self.trainer.log_dir,'delta_output_analysis.csv')
        df.to_csv(csv_path, index=False)
        print('Saved CSV to:', csv_path)

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
    _ = GraceLitModule(None, None, None)
