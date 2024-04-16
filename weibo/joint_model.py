import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils.BaseModel import JointLogitsBaseModel

from transformers import BertModel
from transformers import BertConfig
from torch.optim.lr_scheduler import StepLR

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

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class FusionNet(nn.Module):
    def __init__(
            self, 
            num_classes, 
            loss_fn
            ):
        super(FusionNet, self).__init__()

        self.num_classes = num_classes
        self.loss_fn = loss_fn
        # Create a BERT configuration
        config = BertConfig(
            vocab_size=30522, 
            hidden_size=128, 
            num_hidden_layers=1, 
            num_attention_heads=1, 
            intermediate_size=128, 
            hidden_act='gelu', 
            hidden_dropout_prob=0.1, 
            attention_probs_dropout_prob=0.1, 
            max_position_embeddings=512, 
            type_vocab_size=2, 
            initializer_range=0.02, 
            layer_norm_eps=1e-12, 
            pad_token_id=0,
        )

        self.text_model = BertModel(config)
        self.image_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        for param in self.text_model.parameters():
            param.requires_grad = False
        for param in self.image_backbone.parameters():
            param.requires_grad = False
        self.x2_model = MLP(input_dim=25088, hidden_dim=512, num_classes=num_classes)
        self.x1_model = MLP(input_dim=128, hidden_dim=64, num_classes=num_classes)

        self.w1 = 1.0
        self.w2 = 1.0


    def forward(self, x1_data, x2_data, label):
        """ Forward pass for the FusionNet model. Fuses at logit level.
        
        Args:
            x1_data (torch.Tensor): Input data for modality 1
            x2_data (torch.Tensor): Input data for modality 2
            label (torch.Tensor): Ground truth label

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing the logits for modality 1, modality 2, average logits, and loss
        """


        text_output = mean_pooling(self.text_model(**x1_data), x1_data['attention_mask'])
        image_output = self.image_backbone(x2_data)
        image_output = image_output.view(image_output.size(0), -1)

        x1_logits = self.x1_model(text_output)
        x2_logits = self.x2_model(image_output)

        # fuse at logit level
        avg_logits = (x1_logits + x2_logits) / 2

        loss = self.loss_fn(avg_logits, label)

        return (x1_logits, x2_logits, avg_logits, loss)

class MultimodalWeiboModel(JointLogitsBaseModel): 

    def __init__(self, args): 
        """Initialize MultimodalWeiboModel.

        Args: 
            args (argparse.Namespace): Arguments for the model        
        """

        super(MultimodalWeiboModel, self).__init__(args)

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
            num_classes=self.args.num_classes, 
            loss_fn=nn.CrossEntropyLoss()
        )