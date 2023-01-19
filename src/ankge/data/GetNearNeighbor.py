import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import pickle

class GetNearNeighbor():
    def __init__(self,args,train_triples):
        
        self.args = args
        self.ent_emb = None
        self.rel_emb = None
        
        self.train_triples = train_triples
        self.near_neighbor = None #[num_train, ent_n1 + rel_n2 + tri_ent_n3 + tri_rel_n3]

    def init_emb(self):
        
        model_path = os.path.join(os.getcwd(), 'output', 'link_prediction', self.args.dataset_name, self.args.base_model_name, self.args.base_model_path)
        BaseModel = torch.load(model_path) 
        
        self.ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.ent_emb.weight.data = BaseModel['state_dict']['model.ent_emb.weight']
        self.ent_emb.weight.requires_grad = False

        self.rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        self.rel_emb.weight.data = BaseModel['state_dict']['model.rel_emb.weight']
        self.rel_emb.weight.requires_grad = False

        self.epsilon = 2.0
        self.margin = nn.Parameter(
            torch.Tensor([self.args.margin]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]), 
            requires_grad=False
        )

        if self.args.base_model_name == 'HAKE':
            self.phase_weight = nn.Parameter(
                BaseModel['state_dict']['model.phase_weight'],
                requires_grad=False
            )
            self.modules_weight = nn.Parameter(
                BaseModel['state_dict']['model.modules_weight'],
                requires_grad=False
            )

    def calculate_nearest_neighbor(self):
        
        self.init_emb()

        knn_file_name = [self.args.ent_knn, self.args.rel_knn, self.args.triple_knn, \
                            self.args.triple_ent_knn, self.args.triple_rel_knn, 'kNN.pickle']
        knn_file_name = "_".join([str(knn) for knn in knn_file_name])    
        file = os.path.join(os.getcwd(), 'output', 'kNN_candidate', self.args.dataset_name, self.args.base_model_name, knn_file_name)

        if os.path.exists(file):
            print('using history kNN candidate pickle')
            with open(file, "rb") as handle:
                self.near_neighbor = pickle.load(handle)
            return self.near_neighbor.cpu()

            #self.near_neighbor = torch.randint(10,100, size=(len(self.train_triples), 3))
            #return self.near_neighbor.cpu()

        NNDataloader = DataLoader(
            self.train_triples,
            shuffle=False,
            batch_size=self.args.eval_bs,
            num_workers=self.args.num_workers,
            pin_memory=False,
            drop_last=False,
        )
                
        loop  = tqdm(NNDataloader)
        for data in loop:
            
            loop.set_description("Calculate the nearest neighbor")
            data = torch.stack(data, dim=1).cuda()

            # calculate entity level nearest neighbors
            head_emb, relation_emb, tail_emb = self.tri2emb(triples=data, mode="entity_level")
            score = self.score_func(head_emb, relation_emb, tail_emb)
            self.entity_cand = self.calc_ranks(pred_score=score, topk=self.args.triple_ent_knn).repeat((1, self.args.triple_rel_knn)) #[bs, ent * rel] 
            entity_level = self.calc_ranks(pred_score=score, topk=self.args.ent_knn)  # [bs, ent_n1]

            # calculate relation level nearest neighbors
            head_emb, relation_emb, tail_emb = self.tri2emb(triples=data, mode="relation_level")
            score = self.score_func(head_emb, relation_emb, tail_emb)
            self.relation_cand = self.calc_ranks(pred_score=score, topk=self.args.triple_rel_knn).repeat_interleave(self.args.triple_ent_knn, dim=1) #[bs, rel * ent] 
            relation_level = self.calc_ranks(pred_score=score, topk=self.args.rel_knn) # [bs, rel_n2]

            # calculate triple level nearest neighbors        
            head_emb, relation_emb, tail_emb = self.tri2emb(triples=data, mode='triple_level')
            score = self.score_func(head_emb, relation_emb, tail_emb)
            self.triple_cand = self.calc_ranks(pred_score=score, topk=self.args.triple_knn) #[bs, tri_n3]
            a_range = torch.arange(self.triple_cand.size()[0]).unsqueeze(1)
            triple_level = torch.cat(
                (self.entity_cand[a_range, self.triple_cand], self.relation_cand[a_range, self.triple_cand]), dim=1,   
            ) 

            near_neighbor = torch.cat((entity_level, relation_level, triple_level), dim=1)

            if self.near_neighbor != None:
                self.near_neighbor = torch.cat((self.near_neighbor, near_neighbor), dim=0)
            else:
                self.near_neighbor = near_neighbor

            if self.near_neighbor.shape[0] == self.train_triples.__len__():
                with open(file, "wb") as handle:
                    pickle.dump(self.near_neighbor, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return self.near_neighbor.cpu()

    def calc_ranks(self, pred_score, topk=1):
        
        ranks = torch.argsort(pred_score, dim=1, descending=False)  
        return ranks[:, :topk]
    
    def tri2emb(self, triples, mode):

        if mode == "entity_level":

            #增加对于尾实体的过滤步骤
            head_emb = self.ent_emb.weight.data.unsqueeze(0).repeat((triples.shape[0], 1, 1))
            a_range = torch.arange(triples.shape[0])
            head_emb[a_range, triples[:,2], :] = 10000.0

            #head_emb = self.ent_emb.weight.data.unsqueeze(0) #[1, num_ent, dim]
            relation_emb = self.rel_emb(triples[:, 1]).unsqueeze(1) #[bs, 1, dim]
            tail_emb = self.ent_emb(triples[:, 2]).unsqueeze(1) #[bs, 1, dim]
        
        elif mode == "relation_level":
            head_emb = self.ent_emb(triples[:, 0]).unsqueeze(1) #[bs, 1, dim]
            relation_emb = self.rel_emb.weight.data.unsqueeze(0) #[1, num_ent, dim]
            tail_emb = self.ent_emb(triples[:, 2]).unsqueeze(1) #[bs, 1, dim]
        
        elif mode == "triple_level":
            head_emb = self.ent_emb(self.entity_cand) #[bs, ent*rel, dim]
            relation_emb = self.rel_emb(self.relation_cand) #[bs, rel*ent, dim]
            tail_emb = self.ent_emb(triples[:, 2]).unsqueeze(1) #[bs, 1, dim]

        return head_emb, relation_emb, tail_emb
        
    def score_func(self, head_emb, relation_emb, tail_emb=None):
        if self.args.base_model_name == 'TransE':
            return self.transe_score_func(head_emb, relation_emb, tail_emb)
        elif self.args.base_model_name == 'RotatE':
            return self.rotate_score_func(head_emb, relation_emb, tail_emb)
        elif self.args.base_model_name == 'HAKE':
            return self.hake_score_func(head_emb, relation_emb, tail_emb)
        elif self.args.base_model_name == "PairRE":
            return self.pairre_score_func(head_emb, relation_emb, tail_emb)
        else:
            raise ValueError("this model without the AnKnn incremention")

    def transe_score_func(self, head_emb, relation_emb, tail_emb=None):
        
        if tail_emb == None:
            score = head_emb + relation_emb  #[bs, level_n, dim]
        else:
            score = (head_emb + relation_emb) - tail_emb #[bs, level_n, dim]
            score = torch.norm(score, p=1, dim=-1) #[bs, level_n]
        return score

    def rotate_score_func(self, head_emb, relation_emb, tail_emb=None):
        
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head_emb, 2, dim=-1)

        phase_relation = relation_emb/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        if tail_emb == None:
            score = torch.cat([re_score, im_score], dim=1) #[bs_score, dim]
        else:
            re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)
            re_score = re_score - re_tail
            im_score = im_score - im_tail
            score = torch.stack([re_score, im_score], dim = 0) #[2, bs_score, dim/2]
            score = score.norm(dim = 0)
            score = score.sum(dim = -1)
        return score

    def hake_score_func(self, head_emb, relation_emb, tail_emb=None):

        pi = 3.14159265358979323846
        phase_head, mod_head = torch.chunk(head_emb, 2, dim=-1)
        phase_rela, mod_rela, bias_rela = torch.chunk(relation_emb, 3, dim=-1)

        phase_head = phase_head / (self.embedding_range.item() / pi)
        phase_rela = phase_rela / (self.embedding_range.item() / pi)

        if tail_emb == None:           
            phase_score = (phase_head + phase_rela)
            
            mod_rela = torch.abs(mod_rela)
            bias_rela = torch.clamp(bias_rela, max=1)
            
            indicator = (bias_rela < -mod_rela)
            bias_rela[indicator] = -mod_rela[indicator]
            
            r_score = mod_head * (mod_rela + bias_rela)
            phase_score = torch.abs(torch.sin(phase_score /2)) * self.phase_weight
            r_score = r_score * self.modules_weight
            score = torch.cat([phase_score, r_score], dim=1)
        
        else:    
            phase_tail, mod_tail = torch.chunk(tail_emb, 2, dim=-1)
            phase_tail = phase_tail / (self.embedding_range.item() / pi)
            
            phase_score = (phase_head + phase_rela) - phase_tail
            
            mod_rela = torch.abs(mod_rela)
            bias_rela = torch.clamp(bias_rela, max=1)
            
            indicator = (bias_rela < -mod_rela)
            bias_rela[indicator] = -mod_rela[indicator]
            
            r_score = mod_head * (mod_rela + bias_rela) - mod_tail * (1 - bias_rela)
            phase_score = torch.sum(torch.abs(torch.sin(phase_score /2)), dim=2) * self.phase_weight
            r_score = torch.norm(r_score, dim=2) * self.modules_weight
            score = phase_score + r_score
        return score

    def pairre_score_func(self, head_emb, relation_emb, tail_emb=None):
        re_head, re_tail = torch.chunk(relation_emb, 2, dim=-1)

        head = F.normalize(head_emb, 2, -1)
        
        if tail_emb == None:
            score = head * re_head
            return score
        
        else:
            tail = F.normalize(tail_emb, 2, -1)
            score = head * re_head - tail * re_tail
            return torch.norm(score, p=1, dim=-1)


