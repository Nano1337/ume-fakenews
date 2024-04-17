import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils.BaseModel import QMFBaseModel

from existing_algos.QMF import QMF

from transformers import ChineseCLIPModel
from torch.optim.lr_scheduler import StepLR

import warnings
warnings.filterwarnings("ignore")

class MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=2):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)
    
class FusionNet(nn.Module):
    def __init__(
            self, 
            args,
            loss_fn
            ):
        super(FusionNet, self).__init__()

        self.args = args
        self.num_classes = self.args.num_classes
        self.num_modality = 2
        self.qmf = QMF(self.num_modality, self.args.num_samples)


        self.loss_fn = loss_fn
        self.model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        self.x1_model = MLP(input_dim=512, hidden_dim=50, num_classes=self.args.num_classes)  
        self.x2_model = MLP(input_dim=512, hidden_dim=50, num_classes=self.args.num_classes)

        self.w1 = 1.0
        self.w2 = 1.0

    def forward(self, input, label, idx):
        """ Forward pass for the FusionNet model. Fuses at logit level.
        
        Args:
            x1_data (torch.Tensor): Input data for modality 1
            x2_data (torch.Tensor): Input data for modality 2
            label (torch.Tensor): Ground truth label

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the logits for modality 1, modality 2, average logits, and loss
        """

        output = self.model(**input)

        x1_logits = self.x1_model(output['text_embeds'])
        x2_logits = self.x2_model(output['image_embeds'])

        x1_logits = self.w1 * x1_logits
        x2_logits = self.w2 * x2_logits

        out = torch.stack([x1_logits, x2_logits])
        logits_df, conf = self.qmf.df(out) # logits_df is (B, C), conf is (M, B)
        loss_uni = []
        for n in range(self.num_modality):
            loss_uni.append(self.loss_fn(out[n], label))
            self.qmf.history[n].correctness_update(idx, loss_uni[n], conf[n].squeeze())

        loss_reg = self.qmf.reg_loss(conf, idx.squeeze())
        loss_joint = self.loss_fn(logits_df, label)

        loss = loss_joint + torch.sum(torch.stack(loss_uni)) + loss_reg

        # fuse at logit level
        avg_logits = (x1_logits + x2_logits) / 2

        return (x1_logits, x2_logits, avg_logits, loss, logits_df)

class MultimodalWeiboModel(QMFBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalWeiboModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalWeiboModel, self).__init__(args)
    def forward(self, input, label, idx): 
        return self.model(input, label, idx)

    def training_step(self, batch, batch_idx): 
        """Training step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss
        
        """

        # Extract modality x1, modality x2, and label from batch
        input, label, idx = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss, logits_df = self.model(input, label, idx)

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
    

    def validation_step(self, batch, batch_idx): 
        """Validation step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        input, label, idx = batch

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss, logits_df = self.model(input, label, idx)

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
    
    def test_step(self, batch, batch_idx):
        """Test step for the model. Logs loss and accuracy.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple containing x1, x2, and label
            batch_idx (int): Index of the batch

        Returns:
            torch.Tensor: Loss

        """

        # Extract modality x1, modality x2, and label from batch
        input, label, idx = batch 

        # Get predictions and loss from model
        x1_logits, x2_logits, avg_logits, loss, logits_df = self.model(input, label, idx)

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
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1.0e-4)
        if self.args.use_scheduler:
            scheduler = {
                'scheduler': StepLR(optimizer, step_size=500, gamma=0.5),
                'interval': 'step',
                'frequency': 1,
            }
            return [optimizer], [scheduler]
            
        return optimizer
    
    def _build_model(self):
        return FusionNet(
            args=self.args, 
            loss_fn=nn.CrossEntropyLoss()
        )