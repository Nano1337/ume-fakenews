import torch
import torch.nn as nn
import numpy as np

class History(object):
    '''
        For QMF

        Taken almost directly from QMF repo
    '''

    def __init__(self, n_data):
        self.correctness = np.zeros((n_data))
        self.confidence = np.zeros((n_data))
        self.max_correctness = 1
        self.use_ema = True 
        self.alpha = 0.1

    # correctness update
    def correctness_update(self, data_idx, correctness, confidence):
        #probs = torch.nn.functional.softmax(output, dim=1)
        #confidence, _ = probs.max(dim=1)
        data_idx = data_idx.cpu().numpy()
        if self.use_ema:
            # added by Jenni to see if this would help
            self.correctness[data_idx] = (1-self.alpha)*self.correctness[data_idx] + self.alpha*correctness.cpu().detach().numpy()
        else:
            self.correctness[data_idx] += correctness.cpu().detach().numpy()
        self.confidence[data_idx] = confidence.cpu().detach().numpy()

    # max correctness update
    def max_correctness_update(self, epoch):
        if epoch > 1:
            self.max_correctness += 1

    # correctness normalize (0 ~ 1) range
    def correctness_normalize(self, data):
        data_min = self.correctness.min()
        #data_max = float(self.max_correctness)
        data_max = float(self.correctness.max())

        return (data - data_min) / (data_max - data_min)

    # get target & margin
    def get_target_margin(self, data_idx1, data_idx2):
        data_idx1 = data_idx1.cpu().numpy()
        data_idx2 = data_idx2.cpu().numpy()
        cum_correctness1 = self.correctness[data_idx1]
        cum_correctness2 = self.correctness[data_idx2]
        # normalize correctness values
        cum_correctness1 = self.correctness_normalize(cum_correctness1)
        cum_correctness2 = self.correctness_normalize(cum_correctness2)
        # make target pair
        n_pair = len(data_idx1)
        target1 = cum_correctness1[:n_pair]
        target2 = cum_correctness2[:n_pair]

        # calc target
        greater = np.array(target1 > target2, dtype='float')
        less = np.array(target1 < target2, dtype='float') * (-1)

        target = greater + less
        target = torch.from_numpy(target).float().cuda()
        # calc margin
        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float().cuda()

        return target, margin

class QMF:
    '''
        Code piecemealed from QMF repo

        Example usage:
        
        In dataloaders:
            add sample idx
        
        In FusionNet.__init__:
            self.qmf = QMF(n_modality, n_train_samples)
            self.num_modality = n_modality

        In FusionNet.forward train pass:
            out = torch.stack([x1_logits, x2_logits])
            logits_df, conf = self.qmf.df(out)
            loss_uni = []
            for n in range(self.num_modality):
                loss_uni.append(self.loss_fn(out[n], label))
                self.qmf.history[n].correctness_update(idx, loss_uni[n], conf[n].squeeze())

            loss_reg = self.qmf.reg_loss(conf, idx.squeeze())
            loss_joint = self.loss_fn(logits_df, label)

            loss = loss_joint + torch.sum(torch.stack(loss_uni)) + loss_reg

        In FusionNet.forward test pass:
            df_logits, _ = self.qmf.df(torch.stack([x1_logits, x2_logits]))

            Return df_logits & logits and record val/test accuracy using 
            1) logits
            2) logits corrected (though shouldn't actually be used over logits if L_uni is on, as is case in QMF) 
            3) df_logits 
    '''
    def __init__(self, n_modality, n_data):

        self.history = [History(n_data) for _ in range(n_modality)]
        self.n_modality = n_modality

    def df(self, logits):
        '''
        logits: tensor (num_modality, batch, num_class)
        '''
        energy = torch.log(torch.sum(torch.exp(logits), dim=-1))
        conf = energy / 10
        logits_df = logits * conf.unsqueeze(-1).detach()

        return torch.sum(logits_df, dim=0), conf

    def reg_loss(self, confidence, idx):
        '''
            confidence: tensor (num_modality, len(idx))
        '''
        # make input pair
        rank_input1 = confidence
        rank_input2 = torch.roll(confidence, -1)
        idx2 = torch.roll(idx, -1)

        ranking_loss = []
        for n in range(self.n_modality):
            # calc target, margin
            rank_target, rank_margin = self.history[n].get_target_margin(idx, idx2)
            rank_target_nonzero = rank_target.clone()
            rank_target_nonzero[rank_target_nonzero == 0] = 1
            rank_input2 = rank_input2[n] + (rank_margin[n] / rank_target_nonzero).reshape((-1,1))
            # ranking loss
            rl = nn.MarginRankingLoss(margin=0.0)(rank_input1[n],
                                        rank_input2[n],
                                        -rank_target)#.reshape(-1,1))
            ranking_loss.append(rl)

        return torch.sum(torch.stack(ranking_loss))
