import torch 
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from utils.BaseModel import QMFBaseModel
from existing_algos.QMF import QMF

from torch.optim.lr_scheduler import StepLR

class MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=101):
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
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        for param in self.model.parameters():
            param.requires_grad = True
        self.x1_model = MLP(input_dim=768, hidden_dim=512, num_classes=self.num_classes)
        self.x2_model = MLP(input_dim=768, hidden_dim=512, num_classes=self.num_classes)


    def forward(self, x1_data, x2_data, label, idx):
        """ Forward pass for the FusionNet model. Fuses at logit level.
        
        Args:
            x1_data (torch.Tensor): Input data for modality 1
            x2_data (torch.Tensor): Input data for modality 2
            label (torch.Tensor): Ground truth label

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the logits for modality 1, modality 2, average logits, and loss
        """

        output = self.model(x1_data, x2_data)
    
        x1_logits = self.x1_model(output['text_embeds'])
        x2_logits = self.x2_model(output['image_embeds'])

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

class MultimodalFakedditModel(QMFBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalFakedditModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalFakedditModel, self).__init__(args)

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