from abc import ABC, abstractmethod
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from utils.EMA import EMA

# existing benchmark algos
from existing_algos.OGM_GE import ogm_ge


class JointLogitsBaseModel(pl.LightningModule, ABC): 

    def __init__(self, args): 
        """Initialize JointLogitsBaseModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(JointLogitsBaseModel, self).__init__()

        self.args = args
        self.model = self._build_model()

        self.num_modality = 2
        self.ema_offset = EMA(torch.zeros(self.num_modality, self.args.num_classes))

        self.train_metrics = {
            "train_loss": [],
            "train_acc": [],
            "train_logits": [],
            "train_x1_acc_uncal": [],
            "train_x2_acc_uncal": [],
            "train_x1_acc": [],
            "train_x2_acc": [],
        }

        self.val_metrics = {
            "val_loss": [], 
            "val_acc": [],
            "val_logits": [],
            "val_labels": [],
        }

        self.test_metrics = {
            "test_loss": [], 
            "test_acc": [], 
            "test_logits": [],
            "test_labels": [],
        }

    def forward(self, x1, x2, label): 
        return self.model(x1, x2, label)

    def training_step(self, batch, batch_idx): 
        """Training step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss
        
        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss = self.model(x1, x2, label)

        # Calculate uncalibrated accuracy for x1 and x2
        x1_acc_uncal = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc_uncal = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())

        # calibrate unimodal logits
        logits_stack = torch.stack([x1_logits, x2_logits])
        self.ema_offset.update(torch.mean(logits_stack, dim=1))
        x1_logits_cal = x1_logits + self.ema_offset.offset[0].to(x1_logits.get_device())
        x2_logits_cal = x2_logits + self.ema_offset.offset[1].to(x2_logits.get_device())

        # Calculate calibrated accuracy for x1 and x2
        x1_acc_cal = torch.mean((torch.argmax(x1_logits_cal, dim=1) == label).float())
        x2_acc_cal = torch.mean((torch.argmax(x2_logits_cal, dim=1) == label).float())

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("train_step/train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_acc", x1_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_acc", x2_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_uncal_acc", x1_acc_uncal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_uncal_acc", x2_acc_uncal, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # accumulate accuracies and losses
        self.train_metrics["train_acc"].append(joint_acc)
        self.train_metrics["train_loss"].append(loss)
        self.train_metrics["train_x1_acc_uncal"].append(x1_acc_uncal.item())
        self.train_metrics["train_x2_acc_uncal"].append(x2_acc_uncal.item())
        self.train_metrics["train_x1_acc"].append(x1_acc_cal.item())
        self.train_metrics["train_x2_acc"].append(x2_acc_cal.item())


        # Return the loss
        return loss
    
    
    def on_train_epoch_end(self) -> None:
        """ Called at the end of the training epoch. Logs average loss and accuracy.

        """
        avg_loss = torch.stack(self.train_metrics["train_loss"]).mean()
        avg_acc = torch.stack(self.train_metrics["train_acc"]).mean()

        self.log("train_epoch/train_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc_uncal", np.mean(np.array(self.train_metrics["train_x1_acc_uncal"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc_uncal", np.mean(np.array(self.train_metrics["train_x2_acc_uncal"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc", np.mean(np.array(self.train_metrics["train_x1_acc"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc", np.mean(np.array(self.train_metrics["train_x2_acc"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.train_metrics["train_loss"].clear()
        self.train_metrics["train_acc"].clear()
        self.train_metrics["train_x1_acc_uncal"].clear()
        self.train_metrics["train_x2_acc_uncal"].clear()
        self.train_metrics["train_x1_acc"].clear()
        self.train_metrics["train_x2_acc"].clear()

    def validation_step(self, batch, batch_idx): 
        """Validation step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss = self.model(x1, x2, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("val_step/val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_step/val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_logits"].append(torch.stack((x1_logits, x2_logits), dim=1))
        self.val_metrics["val_labels"].append(label)
        self.val_metrics["val_loss"].append(loss)
        self.val_metrics["val_acc"].append(joint_acc)
 
        return loss

    def on_validation_epoch_end(self) -> None:
        """ Called at the end of the validation epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        labels = torch.cat(self.val_metrics["val_labels"], dim=0) # (N)
        logits = torch.cat(self.val_metrics["val_logits"], dim=0) # (N, M, C)
        m_out = torch.mean(logits, dim=0)
        offset = torch.mean(m_out, dim=0, keepdim=True) - m_out # (M, C)
        corrected_logits = logits + offset

        x1_logits_uncal = logits[:, 0, :]
        x2_logits_uncal = logits[:, 1, :]
        x1_logits = corrected_logits[:, 0, :]
        x2_logits = corrected_logits[:, 1, :]

        x1_acc_uncal = torch.mean((torch.argmax(x1_logits_uncal, dim=1) == labels).float())
        x2_acc_uncal = torch.mean((torch.argmax(x2_logits_uncal, dim=1) == labels).float())
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == labels).float())
        avg_loss = torch.stack(self.val_metrics["val_loss"]).mean()
        avg_acc = torch.stack(self.val_metrics["val_acc"]).mean()

        self.log("val_epoch/val_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x1_acc_uncal", x1_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x2_acc_uncal", x2_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.val_metrics["val_loss"].clear()
        self.val_metrics["val_acc"].clear()
        self.val_metrics["val_logits"].clear()
        self.val_metrics["val_labels"].clear()


    def test_step(self, batch, batch_idx):
        """Test step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = batch 

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss = self.model(x1, x2, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("test_step/test_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_step/test_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_logits"].append(torch.stack((x1_logits, x2_logits), dim=1))
        self.test_metrics["test_labels"].append(label)
        self.test_metrics["test_loss"].append(loss)
        self.test_metrics["test_acc"].append(joint_acc)

        # Return the loss
        return loss
    
    def on_test_epoch_end(self):
        """ Called at the end of the test epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        labels = torch.cat(self.test_metrics["test_labels"], dim=0) # (N)
        logits = torch.cat(self.test_metrics["test_logits"], dim=0) # (N, M, C)
        m_out = torch.mean(logits, dim=0)
        offset = torch.mean(m_out, dim=0, keepdim=True) - m_out # (M, C)
        corrected_logits = logits + offset

        x1_logits_uncal = logits[:, 0, :]
        x2_logits_uncal = logits[:, 1, :]
        x1_logits = corrected_logits[:, 0, :]
        x2_logits = corrected_logits[:, 1, :]

        x1_acc_uncal = torch.mean((torch.argmax(x1_logits_uncal, dim=1) == labels).float())
        x2_acc_uncal = torch.mean((torch.argmax(x2_logits_uncal, dim=1) == labels).float())
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == labels).float())
        avg_loss = torch.stack(self.test_metrics["test_loss"]).mean()
        avg_accuracy = torch.stack(self.test_metrics["test_acc"]).mean()
        
        self.log("test_epoch/test_avg_acc", avg_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True) 
        self.log("test_epoch/test_avg_x1_acc_uncal", x1_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x2_acc_uncal", x2_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_loss"].clear()
        self.test_metrics["test_acc"].clear()
        self.test_metrics["test_logits"].clear()
        self.test_metrics["test_labels"].clear()

    # Required for pl.LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1.0e-4)
        if self.args.use_scheduler:
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=70, gamma=0.1),
                'interval': 'epoch',
                'frequency': 1,
            }
            return [optimizer], [scheduler]
            
        return optimizer

    @abstractmethod
    def _build_model(self):
        pass

class EnsembleBaseModel(pl.LightningModule, ABC): 
    
    def __init__(self, args): 
        """Initialize EnsembleBaseModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(EnsembleBaseModel, self).__init__()

        self.args = args
        self.model = self._build_model()

        self.train_metrics = {
            "train_loss": [],
            "train_acc": [],
            "train_x1_acc": [],
            "train_x2_acc": [],
        }

        self.val_metrics = {
            "val_loss": [], 
            "val_acc": [],
            "val_x1_acc": [], 
            "val_x2_acc": [],
        }

        self.test_metrics = {
            "test_loss": [], 
            "test_acc": [], 
            "test_x1_acc": [],
            "test_x2_acc": [],
        }

    def forward(self, x1, x2, label): 
        return self.model(x1, x2, label)

    def training_step(self, batch, batch_idx): 
        """Training step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss
        
        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, x1_loss, x2_loss = self.model(x1, x2, label)

        # Calculate acc, unimodal acc not uncalibrated
        x1_acc_cal = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc_cal = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())
        avg_logits = (x1_logits + x2_logits) / 2
        preds = torch.argmax(avg_logits, dim=1)
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = (x1_loss + x2_loss) 

        # Log loss and accuracy
        self.log("train_step/train_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_acc", x1_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_acc", x2_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # accumulate accuracies and losses
        self.train_metrics["train_loss"].append(avg_loss)
        self.train_metrics["train_acc"].append(joint_acc)
        self.train_metrics["train_x1_acc"].append(x1_acc_cal)
        self.train_metrics["train_x2_acc"].append(x2_acc_cal)

        # Return the loss
        return avg_loss
    
    def on_train_epoch_end(self) -> None:
        """ Called at the end of the training epoch. Logs average loss and accuracy.

        """
        avg_loss = torch.stack(self.train_metrics["train_loss"]).mean()
        avg_acc = torch.stack(self.train_metrics["train_acc"]).mean()
        x1_acc = torch.mean(torch.stack(self.train_metrics["train_x1_acc"]))
        x2_acc = torch.mean(torch.stack(self.train_metrics["train_x2_acc"]))

        self.log("train_epoch/train_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.train_metrics["train_loss"].clear()
        self.train_metrics["train_acc"].clear()
        self.train_metrics["train_x1_acc"].clear()
        self.train_metrics["train_x2_acc"].clear()

    def validation_step(self, batch, batch_idx): 
        """Validation step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, x1_loss, x2_loss = self.model(x1, x2, label)

        # Calculate accuracy
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())
        avg_logits = (x1_logits + x2_logits) / 2
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = (x1_loss + x2_loss) / 2

        # Log loss and accuracy
        self.log("val_step/val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_step/val_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_loss"].append(avg_loss)
        self.val_metrics["val_acc"].append(joint_acc)
        self.val_metrics["val_x1_acc"].append(x1_acc)
        self.val_metrics["val_x2_acc"].append(x2_acc)

        # Return the loss
        return avg_loss

    def on_validation_epoch_end(self) -> None:
        """ Called at the end of the validation epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        avg_loss = torch.stack(self.val_metrics["val_loss"]).mean()
        avg_acc = torch.stack(self.val_metrics["val_acc"]).mean()
        x1_acc = torch.mean(torch.stack(self.val_metrics["val_x1_acc"]))
        x2_acc = torch.mean(torch.stack(self.val_metrics["val_x2_acc"]))

        self.log("val_epoch/val_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.val_metrics["val_loss"].clear()
        self.val_metrics["val_acc"].clear()
        self.val_metrics["val_x1_acc"].clear()
        self.val_metrics["val_x2_acc"].clear()

    def test_step(self, batch, batch_idx):
        """Test step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, x1_loss, x2_loss = self.model(x1, x2, label)

        # Calculate accuracy
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())
        avg_logits = (x1_logits + x2_logits) / 2
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        avg_loss = (x1_loss + x2_loss) / 2

        # Log loss and accuracy
        self.log("test_step/test_loss", avg_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_step/test_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_loss"].append(avg_loss)
        self.test_metrics["test_acc"].append(joint_acc)
        self.test_metrics["test_x1_acc"].append(x1_acc)
        self.test_metrics["test_x2_acc"].append(x2_acc)

        # Return the loss
        return avg_loss
    
    def on_test_epoch_end(self):
        """ Called at the end of the test epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        avg_loss = torch.stack(self.test_metrics["test_loss"]).mean()
        avg_acc = torch.stack(self.test_metrics["test_acc"]).mean()
        x1_acc = torch.mean(torch.stack(self.test_metrics["test_x1_acc"]))
        x2_acc = torch.mean(torch.stack(self.test_metrics["test_x2_acc"]))

        self.log("test_epoch/test_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.test_metrics["test_loss"].clear()
        self.test_metrics["test_acc"].clear()
        self.test_metrics["test_x1_acc"].clear()
        self.test_metrics["test_x2_acc"].clear()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1.0e-4)
        if self.args.use_scheduler:
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=70, gamma=0.1),
                'interval': 'epoch',
                'frequency': 1,
            }
            return [optimizer], [scheduler]
            
        return optimizer

    @abstractmethod
    def _build_model(self):
        pass

class JointProbLogitsBaseModel(pl.LightningModule, ABC): 

    def __init__(self, args): 
        """Initialize JointProbLogitsBaseModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """


        super(JointProbLogitsBaseModel, self).__init__()

        self.args = args
        self.model = self._build_model()

        self.num_modality = 2
        self.ema_offset = EMA(torch.zeros(self.num_modality, self.args.num_classes))

        self.train_metrics = {
            "train_loss": [],
            "train_acc": [],
            "train_logits": [],
            "train_x1_acc_uncal": [],
            "train_x2_acc_uncal": [],
            "train_x1_acc": [],
            "train_x2_acc": [],
        }

        self.val_metrics = {
            "val_loss": [], 
            "val_acc": [],
            "val_logits": [],
            "val_labels": [],   
        }

        self.test_metrics = {
            "test_loss": [], 
            "test_acc": [], 
            "test_logits": [],
            "test_labels": [],
        }


    def forward(self, x1, x2, label): 
        return self.model(x1, x2, label)

    def training_step(self, batch, batch_idx): 
        """Training step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss
        
        """
        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = batch

        # Get predictions and loss from model
        x1_logprobs, x2_logprobs, avg_logprobs, loss = self.model(x1, x2, label)

        # Calculate uncalibrated accuracy for x1 and x2
        x1_acc_uncal = torch.mean((torch.argmax(x1_logprobs, dim=1) == label).float())
        x2_acc_uncal = torch.mean((torch.argmax(x2_logprobs, dim=1) == label).float())

        # calibrate unimodal logits
        logprobs_stack = torch.stack([x1_logprobs, x2_logprobs])
        self.ema_offset.update(torch.mean(logprobs_stack, dim=1))
        x1_logprobs_cal = x1_logprobs + self.ema_offset.offset[0].to(x1_logprobs.get_device())
        x2_logprobs_cal = x2_logprobs + self.ema_offset.offset[1].to(x2_logprobs.get_device())

        # Calculate calibrated accuracy for x1 and x2
        x1_acc_cal = torch.mean((torch.argmax(x1_logprobs_cal, dim=1) == label).float())
        x2_acc_cal = torch.mean((torch.argmax(x2_logprobs_cal, dim=1) == label).float())

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logprobs, dim=1) == label).float())

        # Log loss and accuracy
        self.log("train_step/train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_acc", x1_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_acc", x2_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_uncal_acc", x1_acc_uncal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_uncal_acc", x2_acc_uncal, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # accumulate accuracies and losses
        self.train_metrics["train_acc"].append(joint_acc)
        self.train_metrics["train_loss"].append(loss)
        self.train_metrics["train_x1_acc_uncal"].append(x1_acc_uncal.item())
        self.train_metrics["train_x2_acc_uncal"].append(x2_acc_uncal.item())
        self.train_metrics["train_x1_acc"].append(x1_acc_cal.item())
        self.train_metrics["train_x2_acc"].append(x2_acc_cal.item())


        # Return the loss
        return loss

    def on_train_epoch_end(self) -> None:
        """ Called at the end of the training epoch. Logs average loss and accuracy.

        """
        avg_loss = torch.stack(self.train_metrics["train_loss"]).mean()
        avg_acc = torch.stack(self.train_metrics["train_acc"]).mean()

        self.log("train_epoch/train_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc_uncal", np.mean(np.array(self.train_metrics["train_x1_acc_uncal"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc_uncal", np.mean(np.array(self.train_metrics["train_x2_acc_uncal"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc", np.mean(np.array(self.train_metrics["train_x1_acc"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc", np.mean(np.array(self.train_metrics["train_x2_acc"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.train_metrics["train_loss"].clear()
        self.train_metrics["train_acc"].clear()
        self.train_metrics["train_x1_acc_uncal"].clear()
        self.train_metrics["train_x2_acc_uncal"].clear()
        self.train_metrics["train_x1_acc"].clear()
        self.train_metrics["train_x2_acc"].clear()

    def validation_step(self, batch, batch_idx): 
        """Validation step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss = self.model(x1, x2, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("val_step/val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_step/val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_logits"].append(torch.stack((x1_logits, x2_logits), dim=1))
        self.val_metrics["val_labels"].append(label)
        self.val_metrics["val_loss"].append(loss)
        self.val_metrics["val_acc"].append(joint_acc)

        return loss

    def on_validation_epoch_end(self) -> None:
        """ Called at the end of the validation epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        labels = torch.cat(self.val_metrics["val_labels"], dim=0) # (N)
        logits = torch.cat(self.val_metrics["val_logits"], dim=0) # (N, M, C)
        m_out = torch.mean(logits, dim=0)
        offset = torch.mean(m_out, dim=0, keepdim=True) - m_out # (M, C)
        corrected_logits = logits + offset

        x1_logits_uncal = logits[:, 0, :]
        x2_logits_uncal = logits[:, 1, :]
        x1_logits = corrected_logits[:, 0, :]
        x2_logits = corrected_logits[:, 1, :]

        x1_acc_uncal = torch.mean((torch.argmax(x1_logits_uncal, dim=1) == labels).float())
        x2_acc_uncal = torch.mean((torch.argmax(x2_logits_uncal, dim=1) == labels).float())
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == labels).float())
        avg_loss = torch.stack(self.val_metrics["val_loss"]).mean()
        avg_acc = torch.stack(self.val_metrics["val_acc"]).mean()

        self.log("val_epoch/val_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x1_acc_uncal", x1_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x2_acc_uncal", x2_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.val_metrics["val_loss"].clear()
        self.val_metrics["val_acc"].clear()
        self.val_metrics["val_logits"].clear()
        self.val_metrics["val_labels"].clear()


    def test_step(self, batch, batch_idx):
        """Test step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = batch 

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss = self.model(x1, x2, label)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("test_step/test_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_step/test_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_logits"].append(torch.stack((x1_logits, x2_logits), dim=1))
        self.test_metrics["test_labels"].append(label)
        self.test_metrics["test_loss"].append(loss)
        self.test_metrics["test_acc"].append(joint_acc)

        # Return the loss
        return loss
    
    def on_test_epoch_end(self):
        """ Called at the end of the test epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        labels = torch.cat(self.test_metrics["test_labels"], dim=0) # (N)
        logits = torch.cat(self.test_metrics["test_logits"], dim=0) # (N, M, C)
        m_out = torch.mean(logits, dim=0)
        offset = torch.mean(m_out, dim=0, keepdim=True) - m_out # (M, C)
        corrected_logits = logits + offset

        x1_logits_uncal = logits[:, 0, :]
        x2_logits_uncal = logits[:, 1, :]
        x1_logits = corrected_logits[:, 0, :]
        x2_logits = corrected_logits[:, 1, :]

        x1_acc_uncal = torch.mean((torch.argmax(x1_logits_uncal, dim=1) == labels).float())
        x2_acc_uncal = torch.mean((torch.argmax(x2_logits_uncal, dim=1) == labels).float())
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == labels).float())
        avg_loss = torch.stack(self.test_metrics["test_loss"]).mean()
        avg_accuracy = torch.stack(self.test_metrics["test_acc"]).mean()

        self.log("test_epoch/test_avg_acc", avg_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x1_acc_uncal", x1_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x2_acc_uncal", x2_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_loss"].clear()
        self.test_metrics["test_acc"].clear()
        self.test_metrics["test_logits"].clear()
        self.test_metrics["test_labels"].clear()

    # Required for pl.LightningModule
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1.0e-4)
        if self.args.use_scheduler:
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=70, gamma=0.1),
                'interval': 'epoch',
                'frequency': 1,
            }
            return [optimizer], [scheduler]
            
        return optimizer

    @abstractmethod
    def _build_model(self):
        pass

class OGMGEBaseModel(JointLogitsBaseModel, ABC): 

    def __init__(self, args): 
        """Initialize MultimodalCremadModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """


        super(OGMGEBaseModel, self).__init__(args)

        self.automatic_optimization = False
        self.ogm_modulation = self.args.grad_mod_type
        self.ogm_alpha = self.args.alpha


    def forward(self, x1, x2, label): 
        return self.model(x1, x2, label)

    # override to apply OGM-GE gradient modulation
    def training_step(self, batch, batch_idx): 
        """Training step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss
        
        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss = self.model(x1, x2, label)

        # Calculate uncalibrated accuracy for x1 and x2
        x1_acc_uncal = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc_uncal = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())

        # calibrate unimodal logits
        logits_stack = torch.stack([x1_logits, x2_logits])
        self.ema_offset.update(torch.mean(logits_stack, dim=1))
        x1_logits_cal = x1_logits + self.ema_offset.offset[0].to(x1_logits.get_device())
        x2_logits_cal = x2_logits + self.ema_offset.offset[1].to(x2_logits.get_device())

        # Calculate calibrated accuracy for x1 and x2
        x1_acc_cal = torch.mean((torch.argmax(x1_logits_cal, dim=1) == label).float())
        x2_acc_cal = torch.mean((torch.argmax(x2_logits_cal, dim=1) == label).float())

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())

        # Log loss and accuracy
        self.log("train_step/train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_acc", x1_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_acc", x2_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_uncal_acc", x1_acc_uncal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_uncal_acc", x2_acc_uncal, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # accumulate accuracies and losses
        self.train_metrics["train_acc"].append(joint_acc)
        self.train_metrics["train_loss"].append(loss)
        self.train_metrics["train_x1_acc_uncal"].append(x1_acc_uncal.item())
        self.train_metrics["train_x2_acc_uncal"].append(x2_acc_uncal.item())
        self.train_metrics["train_x1_acc"].append(x1_acc_cal.item())
        self.train_metrics["train_x2_acc"].append(x2_acc_cal.item())

        # apply gradient modulatiaon using OGM-GE
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        if self.ogm_modulation:
            ogm_ge(self.model, x1_logits, x2_logits, label, modulation=self.ogm_modulation, alpha=self.ogm_alpha)
        opt.step()

        # Return the loss
        return loss

    # override to manually step lr scheduler
    def on_train_epoch_end(self) -> None:
        """ Called at the end of the training epoch. Logs average loss and accuracy.

        """
        avg_loss = torch.stack(self.train_metrics["train_loss"]).mean()
        avg_acc = torch.stack(self.train_metrics["train_acc"]).mean()

        self.log("train_epoch/train_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc_uncal", np.mean(np.array(self.train_metrics["train_x1_acc_uncal"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc_uncal", np.mean(np.array(self.train_metrics["train_x2_acc_uncal"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc", np.mean(np.array(self.train_metrics["train_x1_acc"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc", np.mean(np.array(self.train_metrics["train_x2_acc"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        
        self.train_metrics["train_loss"].clear()
        self.train_metrics["train_acc"].clear()
        self.train_metrics["train_x1_acc_uncal"].clear()
        self.train_metrics["train_x2_acc_uncal"].clear()
        self.train_metrics["train_x1_acc"].clear()
        self.train_metrics["train_x2_acc"].clear()

        if self.args.use_scheduler:
            schedulers = self.lr_schedulers()
            
            # handle single scheduler and step schedulers per epoch
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
            for scheduler in schedulers:
                scheduler.step()
    @abstractmethod
    def _build_model(self):
        pass

class QMFBaseModel(JointLogitsBaseModel, ABC): 
    
    def __init__(self, args):

        super(QMFBaseModel, self).__init__(args)

        self.train_metrics.update({"train_df_acc": []})
        self.val_metrics.update({"val_df_acc": []})
        self.test_metrics.update({"test_df_acc": []})

    def forward(self, x1, x2, label, idx): 
        return self.model(x1, x2, label, idx)

    def training_step(self, batch, batch_idx): 
        """Training step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss
        
        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label, idx = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss, logits_df = self.model(x1, x2, label, idx)

        # Calculate uncalibrated accuracy for x1 and x2
        x1_acc_uncal = torch.mean((torch.argmax(x1_logits, dim=1) == label).float())
        x2_acc_uncal = torch.mean((torch.argmax(x2_logits, dim=1) == label).float())

        # calibrate unimodal logits
        logits_stack = torch.stack([x1_logits, x2_logits])
        self.ema_offset.update(torch.mean(logits_stack, dim=1))
        x1_logits_cal = x1_logits + self.ema_offset.offset[0].to(x1_logits.get_device())
        x2_logits_cal = x2_logits + self.ema_offset.offset[1].to(x2_logits.get_device())

        # Calculate calibrated accuracy for x1 and x2
        x1_acc_cal = torch.mean((torch.argmax(x1_logits_cal, dim=1) == label).float())
        x2_acc_cal = torch.mean((torch.argmax(x2_logits_cal, dim=1) == label).float())

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        logits_df_acc = torch.mean((torch.argmax(logits_df, dim=1) == label).float())

        # Log loss and accuracy
        self.log("train_step/train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_acc", x1_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_acc", x2_acc_cal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x1_uncal_acc", x1_acc_uncal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_x2_uncal_acc", x2_acc_uncal, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_step/train_df_acc", logits_df_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # accumulate accuracies and losses
        self.train_metrics["train_acc"].append(joint_acc)
        self.train_metrics["train_loss"].append(loss)
        self.train_metrics["train_x1_acc_uncal"].append(x1_acc_uncal.item())
        self.train_metrics["train_x2_acc_uncal"].append(x2_acc_uncal.item())
        self.train_metrics["train_x1_acc"].append(x1_acc_cal.item())
        self.train_metrics["train_x2_acc"].append(x2_acc_cal.item())
        self.train_metrics["train_df_acc"].append(logits_df_acc)


        # Return the loss
        return loss
    
    
    def on_train_epoch_end(self) -> None:
        """ Called at the end of the training epoch. Logs average loss and accuracy.

        """
        avg_loss = torch.stack(self.train_metrics["train_loss"]).mean()
        avg_acc = torch.stack(self.train_metrics["train_acc"]).mean()
        avg_df_acc = torch.mean(torch.stack(self.train_metrics["train_df_acc"]))

        self.log("train_epoch/train_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc_uncal", np.mean(np.array(self.train_metrics["train_x1_acc_uncal"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc_uncal", np.mean(np.array(self.train_metrics["train_x2_acc_uncal"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x1_acc", np.mean(np.array(self.train_metrics["train_x1_acc"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_x2_acc", np.mean(np.array(self.train_metrics["train_x2_acc"])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_epoch/train_avg_df_acc", avg_df_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.train_metrics["train_loss"].clear()
        self.train_metrics["train_acc"].clear()
        self.train_metrics["train_x1_acc_uncal"].clear()
        self.train_metrics["train_x2_acc_uncal"].clear()
        self.train_metrics["train_x1_acc"].clear()
        self.train_metrics["train_x2_acc"].clear()
        self.train_metrics["train_df_acc"].clear()

    def validation_step(self, batch, batch_idx): 
        """Validation step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label, idx = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss, logits_df = self.model(x1, x2, label, idx)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        logits_df_acc = torch.mean((torch.argmax(logits_df, dim=1) == label).float())

        # Log loss and accuracy
        self.log("val_step/val_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_step/logits_df_acc", logits_df_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_step/val_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_logits"].append(torch.stack((x1_logits, x2_logits), dim=1))
        self.val_metrics["val_labels"].append(label)
        self.val_metrics["val_loss"].append(loss)
        self.val_metrics["val_acc"].append(joint_acc)
        self.val_metrics["val_df_acc"].append(logits_df_acc)
 
        return loss

    def on_validation_epoch_end(self) -> None:
        """ Called at the end of the validation epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        labels = torch.cat(self.val_metrics["val_labels"], dim=0) # (N)
        logits = torch.cat(self.val_metrics["val_logits"], dim=0) # (N, M, C)
        m_out = torch.mean(logits, dim=0)
        offset = torch.mean(m_out, dim=0, keepdim=True) - m_out # (M, C)
        corrected_logits = logits + offset

        x1_logits_uncal = logits[:, 0, :]
        x2_logits_uncal = logits[:, 1, :]
        x1_logits = corrected_logits[:, 0, :]
        x2_logits = corrected_logits[:, 1, :]

        x1_acc_uncal = torch.mean((torch.argmax(x1_logits_uncal, dim=1) == labels).float())
        x2_acc_uncal = torch.mean((torch.argmax(x2_logits_uncal, dim=1) == labels).float())
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == labels).float())
        avg_loss = torch.stack(self.val_metrics["val_loss"]).mean()
        avg_acc = torch.stack(self.val_metrics["val_acc"]).mean()
        avg_df_acc = torch.mean(torch.stack(self.val_metrics["val_df_acc"]))

        self.log("val_epoch/val_avg_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x1_acc_uncal", x1_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x2_acc_uncal", x2_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_epoch/val_avg_df_acc", avg_df_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.val_metrics["val_loss"].clear()
        self.val_metrics["val_acc"].clear()
        self.val_metrics["val_logits"].clear()
        self.val_metrics["val_labels"].clear()
        self.val_metrics["val_df_acc"].clear()


    def test_step(self, batch, batch_idx):
        """Test step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        x1, x2, label, idx = batch 

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss, logits_df = self.model(x1, x2, label, idx)

        # Calculate accuracy
        joint_acc = torch.mean((torch.argmax(avg_logits, dim=1) == label).float())
        logits_df_acc = torch.mean((torch.argmax(logits_df, dim=1) == label).float())

        # Log loss and accuracy
        self.log("test_step/test_acc", joint_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_step/test_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_step/logits_df_acc", logits_df_acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_logits"].append(torch.stack((x1_logits, x2_logits), dim=1))
        self.test_metrics["test_labels"].append(label)
        self.test_metrics["test_loss"].append(loss)
        self.test_metrics["test_acc"].append(joint_acc)
        self.test_metrics["test_df_acc"].append(logits_df_acc)

        # Return the loss
        return loss
    
    def on_test_epoch_end(self):
        """ Called at the end of the test epoch. Logs average loss and accuracy.

        Applies unimodal offset correction to logits and calculates accuracy for each modality and jointly

        """
        labels = torch.cat(self.test_metrics["test_labels"], dim=0) # (N)
        logits = torch.cat(self.test_metrics["test_logits"], dim=0) # (N, M, C)
        m_out = torch.mean(logits, dim=0)
        offset = torch.mean(m_out, dim=0, keepdim=True) - m_out # (M, C)
        corrected_logits = logits + offset

        x1_logits_uncal = logits[:, 0, :]
        x2_logits_uncal = logits[:, 1, :]
        x1_logits = corrected_logits[:, 0, :]
        x2_logits = corrected_logits[:, 1, :]

        x1_acc_uncal = torch.mean((torch.argmax(x1_logits_uncal, dim=1) == labels).float())
        x2_acc_uncal = torch.mean((torch.argmax(x2_logits_uncal, dim=1) == labels).float())
        x1_acc = torch.mean((torch.argmax(x1_logits, dim=1) == labels).float())
        x2_acc = torch.mean((torch.argmax(x2_logits, dim=1) == labels).float())
        avg_loss = torch.stack(self.test_metrics["test_loss"]).mean()
        avg_accuracy = torch.stack(self.test_metrics["test_acc"]).mean()
        avg_df_acc = torch.mean(torch.stack(self.test_metrics["test_df_acc"]))

        self.log("test_epoch/test_avg_acc", avg_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True) 
        self.log("test_epoch/test_avg_x1_acc_uncal", x1_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x2_acc_uncal", x2_acc_uncal, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x1_acc", x1_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_x2_acc", x2_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_epoch/test_avg_df_acc", avg_df_acc, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.test_metrics["test_loss"].clear()
        self.test_metrics["test_acc"].clear()
        self.test_metrics["test_logits"].clear()
        self.test_metrics["test_labels"].clear()
        self.test_metrics["test_df_acc"].clear()

    @abstractmethod
    def _build_model(self): 
        pass