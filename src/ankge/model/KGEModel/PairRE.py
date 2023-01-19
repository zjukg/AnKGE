from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Model

class PairRE(Model):

    def __init__(self, args):
        super(PairRE, self).__init__(args)
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        self.init_emb()

    def init_emb(self):
        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]),
            requires_grad=False,
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]),
            requires_grad=False,
        )
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        nn.init.uniform_(
            tensor=self.ent_emb.weight.data,
            a = -self.embedding_range.item(),
            b = self.embedding_range.item(),
        )
        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim * 2)
        nn.init.uniform_(
            tensor=self.rel_emb.weight.data,
            a = -self.embedding_range.item(),
            b = self.embedding_range.item(), 
        )

    def score_func(self, head_emb, relation_emb, tail_emb, mode):
        re_head, re_tail = torch.chunk(relation_emb, 2, dim=2)

        head = F.normalize(head_emb, 2, -1)
        tail = F.normalize(tail_emb, 2, -1)

        score = head * re_head - tail  * re_tail
        return self.margin.item() - torch.norm(score, p=1, dim=2)

    def forward(self, triples, negs=None, mode='single'):
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, negs, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return score
    
    def get_score(self, batch, mode):
        triples = batch['positive_sample']
        head_emb, relation_emb, tail_emb = self.tri2emb(triples, mode=mode)
        score = self.score_func(head_emb, relation_emb, tail_emb, mode)
        return score
    