import torch
import torch.nn.functional as F
import torch.nn as nn

class AnKnn_Loss(nn.Module):
    def __init__(self, args, model):
        super(AnKnn_Loss, self).__init__()
        self.args = args
        self.model = model

    def forward(self, pos_score):
        
        sim_score, dis_score, distance = pos_score

        ent_level_loss = self.get_adv_loss(sim_score[0], dis_score[0])
        rel_level_loss = self.get_adv_loss(sim_score[1], dis_score[1])
        triple_level_loss = self.get_adv_loss(sim_score[2], dis_score[2])

        if self.args.set_level == 4:
            loss = distance[:, 0].detach() * ent_level_loss + \
                distance[:, 1].detach() * rel_level_loss + \
                distance[:, 2].detach() * triple_level_loss
        elif self.args.set_level == 1: #without entity level
            loss = distance[:, 0].detach() * rel_level_loss + \
                distance[:, 1].detach() * triple_level_loss
        elif self.args.set_level == 2: #without relation level
            loss = distance[:, 0].detach() * ent_level_loss + \
                distance[:, 1].detach() * triple_level_loss
        elif self.args.set_level == 3: #without triple level
            loss = distance[:, 0].detach() * ent_level_loss + \
                distance[:, 1].detach() * rel_level_loss
        else:
            raise ValueError("Setting error loss function parameters")
        return  loss.mean() + self.get_normalize()

    def get_adv_loss(self, sim_score, dis_score):
        if self.args.anafunc == 'norm':
            loss = -F.logsigmoid(-(self.args.alpha * sim_score)-dis_score) #shape:[bs]
        elif self.args.anafunc == 'cos':
            loss = -F.logsigmoid(self.args.alpha * sim_score + dis_score)
        loss = loss.mean(dim=-1)
        # from IPython import embed;embed();exit()
        return loss

    def get_normalize(self):
        """calculating the regularization.
        """
        regularization = self.args.regularization * (
                self.model.ent_matrix.norm(p = 3)**3 + \
                self.model.rel_matrix.norm(p = 3)**3 + \
                self.model.trans_matrix.norm(p = 3)**3
            )
        return regularization