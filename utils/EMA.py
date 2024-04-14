import torch

class EMA:

    '''
        Example usage:

        In FusionNet.__init__:
            self.ema_offset = EMA(torch.zeros(num_modality, num_classes))

        In FusionNet.forward if istrain=True:
            logits_stack = torch.stack([x1_logits, x2_logits])
            self.ema_offset.update(torch.mean(logits_stack, dim=1))

        In LightningModule training_step:
            x1_logits_corrected = x1_logits + self.model.ema_offset.offset[0].to(x1_logits.get_device())
            x2_logits_corrected = x2_logits + self.model.ema_offset.offset[1].to(x2_logits.get_device())
            x1_acc = torch.mean((torch.argmax(x1_logits_corrected, dim=1) == label).float())
            x2_acc = torch.mean((torch.argmax(x2_logits_corrected, dim=1) == label).float())
            self.train_metrics['acc_1'].append(x1_acc.item())
            self.train_metrics['acc_2'].append(x2_acc.item())
    '''

    def __init__(self, x0, smoothing=0.05):
        self.x = x0
        self.smoothing = smoothing
        self.counter = 0

    def update(self, x_new):
        with torch.no_grad():
            #beta = (self.smoothing / (self.counter + 1))
            beta = self.smoothing
            self.x = x_new.detach().cpu() * (beta) + self.x * (1.0 - beta)
            self.counter += 1

    @property
    def offset(self):
        return torch.mean(self.x, dim=0, keepdim=True) - self.x