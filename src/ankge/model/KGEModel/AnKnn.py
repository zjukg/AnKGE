import os
from requests import request
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import Model

class AnKnn(Model):
    def __init__(self,args):
        super(AnKnn,self).__init__(args)
        self.args = args
        
        # base model embedding
        self.base_ent_emb = None
        self.base_rel_emb = None

        # Analogy function
        self.ent_matrix = None
        self.rel_matrix = None
        self.trans_matrix = None
        self.model_path = None

        self.init_emb()

    def init_emb(self):

        # init base model 
        self.model_path = os.path.join(os.getcwd(), 'output', 'link_prediction', self.args.dataset_name, self.args.base_model_name, self.args.base_model_path)
        BaseModel = torch.load(self.model_path)
        
        self.base_ent_emb = nn.Embedding(self.args.num_ent, self.args.emb_dim)
        self.base_ent_emb = nn.Embedding.from_pretrained(BaseModel['state_dict']['model.ent_emb.weight'], freeze=True)

        self.base_rel_emb = nn.Embedding(self.args.num_rel, self.args.emb_dim)
        self.base_rel_emb = nn.Embedding.from_pretrained(BaseModel['state_dict']['model.rel_emb.weight'], freeze=True)

        if self.args.base_model_name == 'TransE':

            # init analogy function
            self.ent_matrix = nn.Parameter(torch.zeros(self.args.num_ent, self.args.emb_dim), requires_grad=True)
            nn.init.normal_(tensor=self.ent_matrix, mean=1, std=0.1,)  #TODO: exp:探索初始化参数
            self.rel_matrix = nn.Parameter(torch.zeros(self.args.num_rel, self.args.emb_dim), requires_grad=True)
            nn.init.normal_(tensor=self.rel_matrix, mean=1, std=0.1,)
            self.trans_matrix = nn.Parameter(torch.zeros(self.args.emb_dim, self.args.emb_dim), requires_grad=True)
            nn.init.normal_(tensor=self.trans_matrix, mean=0, std=0.1,)

        if self.args.base_model_name == 'RotatE':
            self.ent_matrix = nn.Parameter(torch.zeros(self.args.num_ent, self.args.emb_dim * 2 ), requires_grad=True)
            nn.init.normal_(tensor=self.ent_matrix, mean=1, std=0.1,)  #TODO: exp:探索初始化参数
            self.rel_matrix = nn.Parameter(torch.zeros(self.args.num_rel, self.args.emb_dim), requires_grad=True)
            nn.init.normal_(tensor=self.rel_matrix, mean=1, std=0.1,)
            self.trans_matrix = nn.Parameter(torch.zeros(self.args.emb_dim, self.args.emb_dim * 2), requires_grad=True)
            nn.init.normal_(tensor=self.trans_matrix, mean=0, std=0.1,)
            
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
            self.ent_matrix = nn.Parameter(torch.zeros(self.args.num_ent, self.args.emb_dim * 2 ), requires_grad=True)
            nn.init.normal_(tensor=self.ent_matrix, mean=1, std=0.1,)  #TODO: exp:探索初始化参数
            self.rel_matrix = nn.Parameter(torch.zeros(self.args.num_rel, self.args.emb_dim * 3 ), requires_grad=True)
            nn.init.normal_(tensor=self.rel_matrix, mean=1, std=0.1,)
            self.trans_matrix = nn.Parameter(torch.zeros(self.args.emb_dim * 3, self.args.emb_dim * 2), requires_grad=True)
            nn.init.normal_(tensor=self.trans_matrix, mean=0, std=0.1,)

            self.epsilon = 2.0
            self.margin = nn.Parameter(
                torch.Tensor([self.args.margin]), 
                requires_grad=False
            )
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]), 
                requires_grad=False
            )
            self.phase_weight = nn.Parameter(  #TODO 这两个参数是否要学习
                BaseModel['state_dict']['model.phase_weight'],
                requires_grad=False
            )
            self.modules_weight = nn.Parameter(
                BaseModel['state_dict']['model.modules_weight'],
                requires_grad=False
            )
        
        if self.args.base_model_name == 'PairRE':
            self.ent_matrix = nn.Parameter(torch.zeros(self.args.num_ent, self.args.emb_dim), requires_grad=True)
            nn.init.normal_(tensor=self.ent_matrix, mean=1, std=0.1,)  #TODO: exp:探索初始化参数
            self.rel_matrix = nn.Parameter(torch.zeros(self.args.num_rel, self.args.emb_dim * 2), requires_grad=True)
            nn.init.normal_(tensor=self.rel_matrix, mean=1, std=0.1,)
            self.trans_matrix = nn.Parameter(torch.zeros(self.args.emb_dim * 2, self.args.emb_dim), requires_grad=True)
            nn.init.normal_(tensor=self.trans_matrix, mean=0, std=0.1,)

            self.epsilon = 2.0
            self.margin = nn.Parameter(
                torch.Tensor([self.args.margin]), 
                requires_grad=False
            )
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin.item() + self.epsilon) / self.args.emb_dim]), 
                requires_grad=False
            )

    def analogy_func(self, entity=None, relation=None, level=None):

        if level == 'entity_level':
            ent_emb = self.base_ent_emb(entity) #[bs_train, dim]
            ana_ent = self.ent_matrix[entity]   #[bs_train, dim]
            rel_emb = self.base_rel_emb(relation) #[bs_train, dim]
            ana_rel = self.rel_matrix[relation]
            entity_emb = ent_emb * ana_ent #[bs_train, dim]
            trans_emb = torch.mm(rel_emb * ana_rel, self.trans_matrix) #[bs_train, dim]
            return entity_emb + self.args.trans_alpha * trans_emb

        elif level == 'relation_level':
            rel_emb = self.base_rel_emb(relation)
            ana_rel = self.rel_matrix[relation]
            relation_emb = rel_emb * ana_rel #[bs_train, dim]
            return relation_emb

        elif level == 'triple_level':
            triple_ent_emb = self.analogy_func(entity=entity, relation=relation, level='entity_level')
            triple_rel_emb = self.analogy_func(relation=relation, level='relation_level')
            return triple_ent_emb, triple_rel_emb

        else:
            raise ValueError('error level')

    def score_func(self, head_emb, relation_emb, tail_emb=None):
        if self.args.base_model_name == 'TransE':
            return self.transe_score_func(head_emb, relation_emb, tail_emb)
        elif self.args.base_model_name == 'RotatE':
            return self.rotate_score_func(head_emb, relation_emb, tail_emb)
        elif self.args.base_model_name == 'HAKE':
            return self.hake_score_func(head_emb, relation_emb, tail_emb)
        elif self.args.base_model_name == 'PairRE':
            return self.pairre_score_func(head_emb, relation_emb, tail_emb)
        else:
            raise ValueError("this model without the AnKnn incremention")

    def transe_score_func(self, head_emb, relation_emb, tail_emb=None):
        
        if tail_emb == None:
            score = head_emb + relation_emb  #[bs_score, dim]
        else:
            score = (head_emb + relation_emb) - tail_emb #[bs_score, dim]
            score = torch.norm(score, p=1, dim=-1) #[bs_score]
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
            phase_score = torch.sum(torch.abs(torch.sin(phase_score /2)), dim=-1) * self.phase_weight
            r_score = torch.norm(r_score, dim=-1) * self.modules_weight
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

    def complex_score_func(self, head_emb, relation_emb, tail_emb=None):
        re_head, im_head = torch.chunk(head_emb, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation_emb, 2, dim=-1)
        
        if tail_emb == None:
            score = re_head * re_relation + \
                    im_head * re_relation + \
                    re_head * im_relation - \
                    im_head * im_relation - 1
            return score
            
        else:
            re_tail, im_tail = torch.chunk(tail_emb, 2, dim=-1)
            score = re_head * re_tail * re_relation + \
                    im_head * im_tail * re_relation + \
                    re_head * im_tail * im_relation - \
                    im_head * re_tail * im_relation - 1
            return -torch.sum(score, dim=-1)

    def forward(self, pos_sample, entity, relation, triple):
        
        h, r, t = pos_sample.t()

        head_emb = self.base_ent_emb(h)
        rela_emb = self.base_rel_emb(r)
        tail_emb = self.base_ent_emb(t)

        ana_ent_emb = self.analogy_func(entity=h, relation=r, level='entity_level')
        ana_rel_emb = self.analogy_func(relation=r, level='relation_level')
        ana_triple_ent, ana_triple_rel = self.analogy_func(entity=h, relation=r, level='triple_level')
        ana_triple_emb = self.score_func(ana_triple_ent, ana_triple_rel)

        virtual_ent, virtual_rel, virtual_triple_ent, virtual_triple_rel = \
            self.calculate_virtual_embedding(entity, relation, triple, pos_sample)
        virtual_triple = self.score_func(virtual_triple_ent, virtual_triple_rel)
        
        if self.args.anafunc == 'cos':  
            ent_level_score = F.cosine_similarity(ana_ent_emb, virtual_ent, dim=1)  
            rel_level_score = F.cosine_similarity(ana_rel_emb, virtual_rel, dim=1)
            triple_level_score = F.cosine_similarity(ana_triple_emb, virtual_triple, dim=1)

        elif self.args.anafunc == 'norm':
            ent_level_score = torch.norm(ana_ent_emb-virtual_ent, p=2, dim=-1)
            rel_level_score = torch.norm(ana_rel_emb-virtual_rel, p=2, dim=-1)
            triple_level_score = torch.norm(ana_triple_emb-virtual_triple, p=2, dim=-1)

        else:
            raise ValueError("Error anafunc parameters")

        sim_score = (ent_level_score, rel_level_score, triple_level_score)

        ent_level_distance = self.score_func(ana_ent_emb, rela_emb, tail_emb)
        rel_level_distance = self.score_func(head_emb, ana_rel_emb, tail_emb)
        triple_level_distance = self.score_func(ana_triple_ent, ana_triple_rel, tail_emb)

        dis_score = (ent_level_distance, rel_level_distance, triple_level_distance)

        with torch.no_grad():
            ent_level_distance = self.score_func(virtual_ent, rela_emb, tail_emb) #[bs_train]
            rel_level_distance = self.score_func(head_emb, virtual_rel, tail_emb) #[bs_train]
            triple_level_distance = self.score_func(virtual_triple_ent, virtual_triple_rel, tail_emb) #[bs_train]
            base_level_distance = self.score_func(head_emb, rela_emb, tail_emb) #[bs_train]
            
            if self.args.set_loss_distance == 1:
                distance = torch.stack(
                    ( -ent_level_distance, -rel_level_distance, -triple_level_distance, -base_level_distance),
                    dim=1,
                ) #[bs_train, 4]
                distance = F.softmax(distance, dim=1) #[bs_train, 4] 经过softmax
            elif self.args.set_loss_distance == 2:
                all_distance = torch.stack(
                    (ent_level_distance, rel_level_distance, triple_level_distance, base_level_distance),
                    dim=1,
                ) #[bs_train ,4]
                arange_d = torch.arange(all_distance.shape[0])
                argmin_d = torch.argmin(all_distance, dim=1)
                distance = torch.zeros(all_distance.shape).cuda() #[bs_train, 4] min=1 other=0 
                distance[arange_d, argmin_d] = 1
            else:
                raise ValueError("Setting error loss distance function parameter")

        return sim_score, dis_score, distance

    def calculate_virtual_embedding(self, ent_level, rel_level, triple_level, pos_sample):
        
        with torch.no_grad():
            h, r, t = pos_sample.t()
            base_h_emb = self.base_ent_emb(h).unsqueeze(1) #[bs_train, 1 ,embedding]
            base_r_emb = self.base_rel_emb(r).unsqueeze(1)
            base_t_emb = self.base_ent_emb(t).unsqueeze(1)
            # entity level
            
            ent_emb = self.base_ent_emb(ent_level) #[bs_train, entity_knn, embedding]
            e_score = self.score_func(ent_emb, base_r_emb, base_t_emb) #[bs_train, entity_knn]
            virtual_ent = (F.softmax(-e_score, dim=1).unsqueeze(2) * ent_emb).sum(dim=1) #[bs_train, embedding]

            # relation level
            rel_emb = self.base_rel_emb(rel_level)
            r_score = self.score_func(base_h_emb, rel_emb, base_t_emb)
            virtual_rel = (F.softmax(-r_score, dim=1).unsqueeze(2) * rel_emb).sum(dim=1)
            
            # triple level
            triple_ent = self.base_ent_emb(triple_level[:, :self.args.triple_knn]) #[bs_train, triple_knn, embedding]
            triple_rel = self.base_rel_emb(triple_level[:, self.args.triple_knn:]) #[bs_train, triple_knn, embedding]
            t_score = self.score_func(triple_ent, triple_rel, base_t_emb)
            virtual_triple_ent = (F.softmax(-t_score, dim=1).unsqueeze(2) * triple_ent).sum(dim=1)
            virtual_triple_rel = (F.softmax(-t_score, dim=1).unsqueeze(2) * triple_rel).sum(dim=1)  
        
        return virtual_ent, virtual_rel, virtual_triple_ent, virtual_triple_rel #[bs_train, embedding] * 4

    def get_score(self, batch, mode):
        
        triples = batch["positive_sample"]
        if self.args.test_sampler_class == 'RevTestSampler':
            ent_level   = batch["ent_score"] # [eval_bs, num_ent]
            rel_level   = batch["rel_score"]
            triple_level   = batch["triple_score"]
        h,r,t = triples.t()

        head_base_emb = self.base_ent_emb(h).unsqueeze(1)
        relation_base_emb = self.base_rel_emb(r).unsqueeze(1)
        tail_emb = self.base_ent_emb.weight.data.unsqueeze(0)

        head_analogy_emb = self.analogy_func(entity=h, relation=r, level='entity_level').unsqueeze(1)
        rela_analogy_emb = self.analogy_func(relation=r, level='relation_level').unsqueeze(1)

        base_score   = -self.score_func(head_base_emb, relation_base_emb, tail_emb)
        ent_score    = -self.score_func(head_analogy_emb, relation_base_emb, tail_emb)
        rel_score    = -self.score_func(head_base_emb, rela_analogy_emb, tail_emb)
        triple_score = -self.score_func(head_analogy_emb, rela_analogy_emb, tail_emb)

        if self.args.test_sampler_class == 'RevTestSampler':
            return self.args.ent_lambda * ent_level * ent_score + \
                   self.args.rel_lambda * rel_level * rel_score + \
                   self.args.triple_lambda * triple_level * triple_score + \
                   base_score
        else:
            return self.args.ent_lambda * ent_score + \
                   self.args.rel_lambda * rel_score + \
                   self.args.triple_lambda * triple_score + \
                   base_score      

