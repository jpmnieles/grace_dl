from typing import Any, Dict, Optional, Tuple

import os
import numpy as np
import pandas as pd
import copy
import glob

import torch
from lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split


class GraceDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir = "/home/jaynieles/dev/grace_dl/data/thesis/241003_075m_grace_dataset.csv",
        val_size = 0.3,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=True)

        self.val_size = val_size
        self.data_train, self.data_val, self.data_test = self.preprocess_csv(data_dir)
        self.batch_size_per_device = batch_size

    def preprocess_csv(self, data_dir):
        # Read CSV
        df = pd.read_csv(data_dir)

        # Add Cmd of Current Time Step
        df['cmd_lnt_t'] = np.array(len(df)*[-13])
        df['cmd_lnp_t'] = np.array(len(df)*[-40])
        df['cmd_unt_t'] = np.array(len(df)*[44])
        df['cmd_lep_t'] = np.array(len(df)*[-18])
        df['cmd_rep_t'] = np.array(len(df)*[-18])
        df['cmd_et_t'] = np.array(len(df)*[22])

        # Reorganizing DataFrame
        data_df = pd.DataFrame({
            # Input
            'cmd_lnt_tplus1': df['cmd_theta_lower_neck_tilt'].values,
            'cmd_lnp_tplus1': df['cmd_theta_lower_neck_pan'].values,
            'cmd_unt_tplus1': df['cmd_theta_upper_neck_tilt'].values,
            'cmd_et_tplus1': df['cmd_theta_eyes_tilt'].values,
            'cmd_lep_tplus1': df['cmd_theta_left_eye_pan'].values,
            'cmd_rep_tplus1': df['cmd_theta_right_eye_pan'].values,

            # Add Cmd of Current Time Step
            'cmd_lnt_t': np.array(len(df)*[-13]),
            'cmd_lnp_t': np.array(len(df)*[-40]),
            'cmd_unt_t': np.array(len(df)*[44]),
            'cmd_et_t': np.array(len(df)*[22]),
            'cmd_lep_t': np.array(len(df)*[-18]),
            'cmd_rep_t': np.array(len(df)*[-18]),

            # Targets
            'l_rvec_0': df['l_rvec_0'].values,
            'l_rvec_1': df['l_rvec_1'].values,
            'l_rvec_2': df['l_rvec_2'].values,
            'l_tvec_0': df['l_tvec_0'].values,
            'l_tvec_1': df['l_tvec_1'].values,
            'l_tvec_2': df['l_tvec_2'].values,
            'r_rvec_0': df['r_rvec_0'].values,
            'r_rvec_1': df['r_rvec_1'].values,
            'r_rvec_2': df['r_rvec_2'].values,
            'r_tvec_0': df['r_tvec_0'].values,
            'r_tvec_1': df['r_tvec_1'].values,
            'r_tvec_2': df['r_tvec_2'].values,
        })
      
        # Minmax Feature Scaler
        feature_ranges = {
            'cmd_lnt_tplus1': (-44, 44),
            'cmd_lnp_tplus1': (-44, 44),
            'cmd_unt_tplus1': (-44, 44),
            'cmd_et_tplus1': (-44, 44),
            'cmd_lep_tplus1': (-44, 44),
            'cmd_rep_tplus1': (-44, 44),
            'cmd_lnt_t': (-44, 44),
            'cmd_lnp_t': (-44, 44),
            'cmd_unt_t': (-44, 44),
            'cmd_et_t': (-44, 44),
            'cmd_lep_t': (-44, 44),
            'cmd_rep_t': (-44, 44),
            'l_tvec_0': (-1.5, 1.5),  
            'l_tvec_1': (-1.5, 1.5),  
            'l_tvec_2': (-1.5, 1.5),
            'r_tvec_0': (-1.5, 1.5),
            'r_tvec_1': (-1.5, 1.5),  
            'r_tvec_2': (-1.5, 1.5),
        }

        # Create the scaled DataFrame
        scaled_df = data_df.copy()
        for col in data_df[feature_ranges.keys()].columns:
            col_min, col_max = feature_ranges[col]
            scaled_df[col] = 2 * (data_df[col] - col_min) / (col_max - col_min) - 1

        # Separation of Training and Validation Set
        train_df, temp_df = train_test_split(scaled_df, test_size=self.val_size, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=1/3, random_state=42)

        # Training Set
        X_train = torch.tensor(train_df.iloc[:,:12].values, dtype=torch.float32)
        y_train = torch.tensor(train_df.iloc[:,12:].values, dtype=torch.float32)
        train_dataset = TensorDataset(X_train, y_train)

        # Validation Set
        X_val = torch.tensor(val_df.iloc[:,:12].values, dtype=torch.float32)
        y_val = torch.tensor(val_df.iloc[:,12:].values, dtype=torch.float32)
        val_dataset = TensorDataset(X_val, y_val)

        # Test Set
        X_test = torch.tensor(test_df.iloc[:,:12].values, dtype=torch.float32)
        y_test = torch.tensor(test_df.iloc[:,12:].values, dtype=torch.float32)
        test_dataset = TensorDataset(X_test, y_test)

        return train_dataset, val_dataset, test_dataset

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = GraceDataModule()
