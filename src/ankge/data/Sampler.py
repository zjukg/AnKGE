from numpy.random.mtrand import normal
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict as ddict
import random
from .DataPreprocess import *
from IPython import embed
import torch.nn.functional as F
import time
import queue

class UniSampler(BaseSampler):
    """Random negative sampling 
    Filtering out positive samples and selecting some samples randomly as negative samples.

    Attributes:
        cross_sampling_flag: The flag of cross sampling head and tail negative samples.
    """
    def __init__(self, args):
        super().__init__(args)
        self.cross_sampling_flag = 0

    def sampling(self, data):
        """Filtering out positive samples and selecting some samples randomly as negative samples.
        
        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The training data.
        """
        batch_data = {}
        neg_ent_sample = []
        subsampling_weight = []
        self.cross_sampling_flag = 1 - self.cross_sampling_flag
        if self.cross_sampling_flag == 0:
            batch_data['mode'] = "head-batch"
            for h, r, t in data:
                neg_head = self.head_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_head)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r-1)]
                    subsampling_weight.append(weight)
        else:
            batch_data['mode'] = "tail-batch"
            for h, r, t in data:
                neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
                neg_ent_sample.append(neg_tail)
                if self.args.use_weight:
                    weight = self.count[(h, r)] + self.count[(t, -r-1)]
                    subsampling_weight.append(weight)

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_sample'] = torch.LongTensor(np.array(neg_ent_sample))
        if self.args.use_weight:
            batch_data["subsampling_weight"] = torch.sqrt(1/torch.tensor(subsampling_weight))
        return batch_data
    
    def uni_sampling(self, data):
        batch_data = {}
        neg_head_list = []
        neg_tail_list = []
        for h, r, t in data:
            neg_head = self.head_batch(h, r, t, self.args.num_neg)
            neg_head_list.append(neg_head)
            neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
            neg_tail_list.append(neg_tail)

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_head'] = torch.LongTensor(np.arrary(neg_head_list))
        batch_data['negative_tail'] = torch.LongTensor(np.arrary(neg_tail_list))
        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'mode']

class RevUniSampler(RevSampler):
    def __init__(self, args):
        super().__init__(args)

    def sampling(self, data):
        batch_data = {}
        neg_ent_sample = []
        subsampling_weight = []
        for h, r, t in data:
            neg_tail = self.tail_batch(h, r, t, self.args.num_neg)
            neg_ent_sample.append(neg_tail)
            if self.args.use_weight:
                weight = self.count[(h, r)] + self.count[(t, -r-1)]
                subsampling_weight.append(weight)

        batch_data["positive_sample"] = torch.LongTensor(np.array(data))
        batch_data['negative_sample'] = torch.LongTensor(np.array(neg_ent_sample))
        batch_data['mode'] = 'tail-batch'
        if self.args.use_weight:
            batch_data["subsampling_weight"] = torch.sqrt(1/torch.tensor(subsampling_weight))
        return batch_data
        
    def get_sampling_keys(self):
        return ['positive_sample', 'negative_sample', 'mode']

class KnnSampler(RevSampler):
    def __init__(self,args):
        super().__init__(args)
    
    def sampling(self, data):
        batch_data = {}
        section = [3] * 4
        section[1] = section[0] + self.args.ent_knn
        section[2] = section[1] + self.args.rel_knn
        section[3] = section[2] + self.args.triple_knn * 2
        
        data = torch.tensor(data)
        triples = data[:, 0: section[0]]

        # ent_level and negative        
        ent_level = data[:, section[0]: section[1]] #[bs, ent_knn]
        ent_neg   = torch.randint(0, self.args.num_ent, size=(self.args.train_bs, self.args.num_neg))

        # rel_level and negative
        rel_level = data[:, section[1]: section[2]] #[bs, rel_knn]
        rel_neg   = torch.randint(0, self.args.num_rel, size=(self.args.train_bs, self.args.num_neg))

        # triple_level and negative
        triple_level = data[:, section[2]: section[3]] #[bs, triple_knn]
        triple_neg = torch.cat(
                    (torch.randint(0, self.args.num_ent, size=(self.args.train_bs, self.args.num_neg)),
                    torch.randint(0, self.args.num_rel, size=(self.args.train_bs, self.args.num_neg))),
                    dim=1)

        batch_data["positive_sample"] = triples
        batch_data["ent_level"]       = ent_level
        batch_data["ent_neg"]         = ent_neg
        batch_data["rel_level"]       = rel_level
        batch_data["rel_neg"]         = rel_neg
        batch_data["triple_level"]    = triple_level
        batch_data["triple_neg"]      = triple_neg
    
        return batch_data

    def get_sampling_keys(self):
        return ['positive_sample', 'ent_level', 'ent_neg', 'rel_level', 'rel_neg',
                'triple_level', 'triple_neg']

class TestSampler(object):
    """Sampling triples and recording positive triples for testing.

    Attributes:
        sampler: The function of training sampler.
        hr2t_all: Record the tail corresponding to the same head and relation.
        rt2h_all: Record the head corresponding to the same tail and relation.
        num_ent: The count of entities.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        self.num_ent = sampler.args.num_ent

    def get_hr2t_rt2h_from_all(self):
        """Get the set of hr2t and rt2h from all datasets(train, valid, and test), the data type is tensor.

        Update:
            self.hr2t_all: The set of hr2t.
            self.rt2h_all: The set of rt2h.
        """
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))
        for r, t in self.rt2h_all:
            self.rt2h_all[(r, t)] = torch.tensor(list(self.rt2h_all[(r, t)]))

    def sampling(self, data):
        """Sampling triples and recording positive triples for testing.

        Args:
            data: The triples used to be sampled.

        Returns:
            batch_data: The data used to be evaluated.
        """
        batch_data = {}
        head_label = torch.zeros(len(data), self.num_ent)
        tail_label = torch.zeros(len(data), self.num_ent)
        for idx, triple in enumerate(data):
            head, rel, tail = triple
            head_label[idx][self.rt2h_all[(rel, tail)]] = 1.0
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0
        batch_data["positive_sample"] = torch.tensor(data)
        batch_data["head_label"] = head_label
        batch_data["tail_label"] = tail_label
        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "head_label", "tail_label"]

class RevTestSampler(object):

    def __init__(self, sampler):
        self.sampler = sampler
        self.hr2t_all = ddict(set)
        self.get_hr2t_rt2h_from_all()
        
        self.num_ent = sampler.args.num_ent
        self.num_rel = sampler.args.num_rel
        self.rt2h_train = sampler.rt2h_train
        self.ht2r_train = sampler.ht2r_train
        self.t2hr_train = sampler.t2hr_train
        self.init_level()

    def get_hr2t_rt2h_from_all(self):
        self.all_true_triples = self.sampler.get_all_true_triples()
        for h, r, t in self.all_true_triples:
            self.hr2t_all[(h, r)].add(t)
        for h, r in self.hr2t_all:
            self.hr2t_all[(h, r)] = torch.tensor(list(self.hr2t_all[(h, r)]))

    def init_level(self):
        self.ent_level = torch.zeros(self.num_rel, self.num_ent)
        self.rel_level = torch.zeros(self.num_ent, self.num_ent)
        self.triple_level = torch.zeros(self.num_ent)

        for (r, t), score in self.rt2h_train.items():
            self.ent_level[r, t] = len(score)
        for (h, t), score in self.ht2r_train.items():
            self.rel_level[h, t] = len(score)
        for t, score in self.t2hr_train.items():
            self.triple_level[t] = len(score)

    def se_score(self, score, th):
        score = score / th
        ones   = torch.ones(score.shape)
        score = torch.where(score>=1, ones, score)
        return score

    def sampling(self, data):
        batch_data = {}
        tail_label = torch.zeros(len(data), self.num_ent)
        for idx, triple in enumerate(data):
            head, rel, tail = triple
            tail_label[idx][self.hr2t_all[(head, rel)]] = 1.0

        data = torch.tensor(data)
        ent_score = self.ent_level[data[:,1]] # ent_level_score
        ent_score = self.se_score(ent_score, self.sampler.args.ent_knn) #[eval_bs, num_ent]
        rel_score = self.rel_level[data[:,0]] # rel_level_score
        rel_score = self.se_score(rel_score, self.sampler.args.rel_knn)
        triple_score = self.triple_level.repeat(data.shape[0]).reshape(data.shape[0], -1)
        triple_score = self.se_score(triple_score, self.sampler.args.triple_knn)
        
        batch_data["positive_sample"] = data
        batch_data["tail_label"] = tail_label
        batch_data["ent_score"] = ent_score
        batch_data["rel_score"] = rel_score
        batch_data["triple_score"] = triple_score
        return batch_data

    def get_sampling_keys(self):
        return ["positive_sample", "tail_label", "ent_score", "rel_score", "triple_score"]

'''继承torch.Dataset'''
class KGDataset(Dataset):

    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]