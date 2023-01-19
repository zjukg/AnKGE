import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import pickle

class GetLevelScore():
    def __init__(self, args, triples, rt2h, ht2r, t2hr):
        self.args = args
        self.triples = triples
        self.rt2h_train = rt2h # ent_level
        self.ht2r_train = ht2r # rel_level
        self.t2hr_train = t2hr # triple_level

        self.level_candidate = None
    
    def init_level(self):
        self.ent_level = torch.zeros(self.args.num_rel, self.args.num_ent)
        self.rel_level = torch.zeros(self.args.num_ent, self.args.num_ent)
        self.triple_level = torch.zeros(self.args.num_ent)

        for (r, t), score in self.rt2h_train.items():
            self.ent_level[r, t] = len(score)
        for (h, t), score in self.ht2r_train.items():
            self.rel_level[h, t] = len(score)
        for t, score in self.t2hr_train.items():
            self.triple_level[t] = len(score)
        
    def get_level_score(self):

        self.init_level()
        file = os.path.join(os.getcwd(), 'output', 'LP_candidate', self.args.dataset_name, 'level_score.pickle')
        
        if os.path.exists(file):
            print('using history LP candidate pickle')
            with open(file, "rb") as handle:
                self.level_candidate = pickle.load(handle)
            return self.level_candidate.cpu()

        LPDataloader = DataLoader(
            self.triples,
            shuffle=False,
            batch_size=self.args.eval_bs,
            num_workers=self.args.num_workers,
            pin_memory=False,
            drop_last=False
        )

        loop = tqdm(LPDataloader)
        for data in loop:

            loop.set_description("Calculate the LP candidate number")
            data = torch.stack(data, dim=1)
            
            ent_score = self.ent_level[data[:,1]] # ent_level_score
            rel_score = self.rel_level[data[:,0]] # rel_level_score
            triple_score = self.triple_level.repeat(data.shape[0]).reshape(data.shape[0], -1)
            
            level_candidate = torch.cat((ent_score, rel_score, triple_score), dim=1)

            if self.level_candidate != None:
                self.level_candidate = torch.cat((self.level_candidate, level_candidate), dim=0)
            else:
                self.level_candidate = level_candidate

            if self.level_candidate.shape[0] == self.triples.__len__():
                with open(file, "wb") as handle:
                    pickle.dump(self.level_candidate, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.level_candidate.cpu()