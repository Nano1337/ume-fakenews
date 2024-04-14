import torch
import torch.nn.functional as F 

def ogm_ge(model, out_1, out_2, label, alpha=0.1, modulation='OGM_GE'):
    '''
        Example usage:
        In init:
            self.automatic_optimization = False
            self.ogm_modulation = ogm_modulation (None, 'OGM_GE', 'OGM', 'noise')
            self.ogm_alpha = ogm_alpha (float between 0 and 1)
        In training_step:
            opt = self.optimizers()
            opt.zero_grad()
            self.manual_backward(loss)
            if self.ogm_modulation:
                ogm_ge(self, x1_logits, x2_logits, label, modulation=self.ogm_modulation, alpha=self.ogm_alpha)
            opt.step()

        Note: change if/continue below for different models so that only norms are skipped
    '''
    score_v = torch.sum(torch.stack([F.softmax(out_1, dim=-1)[i][label[i]] for i in range(out_1.size(0))]))
    score_a = torch.sum(torch.stack([F.softmax(out_2, dim=-1)[i][label[i]] for i in range(out_2.size(0))]))

    ratio_v = score_v / score_a
    ratio_a = 1 / ratio_v

    """
    Below is the Eq.(10) in our CVPR paper:
            1 - tanh(alpha * rho_t_u), if rho_t_u > 1
    k_t_u =
            1,                         else
    coeff_u is k_t_u, where t means iteration steps and u is modality indicator, either a or v.
            """

    if ratio_v > 1:
        coeff_v = 1 - F.tanh(alpha * F.relu(ratio_v, inplace=True))
        coeff_a = 1
    else:
        coeff_a = 1 - F.tanh(alpha * F.relu(ratio_a, inplace=True))
        coeff_v = 1

    def add_factor(model, coeff):
        for name, parms in model.named_parameters():
            if len(parms.grad.size()) != 4:
                # TODO: update this! Just want to exclude batch norms
                continue
            layer = str(name).split('.')[1]
            if modulation == 'OGM_GE':  # bug fixed
                parms.grad = parms.grad * coeff + \
                                         torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
            elif modulation == 'OGM':
                parms.grad *= coeff
            elif modulation == 'noise':
                parms.grad = parms.grad + torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)

    add_factor(model.x1_model, coeff_v)
    add_factor(model.x2_model, coeff_a)
