import torch
import torch.nn as nn
from models.base import BPR

class LightGCN(BPR):
    def __init__(self, config, dataset):
        super(LightGCN, self).__init__(config, dataset)
        self.layers = config['layers']
        self.adj_matrix = dataset.adj_matrix
        self.norm_adj_matrix = dataset.norm_adj_matrix
        
    def init_embedding(self, num, dim):
        emb = torch.empty(num, dim, device=self.dataset.device)
        # 여기서 논문에서 권장하는 normal(std=0.1) 초기화
        nn.init.normal_(emb, std=0.1)
        return emb

    def get_embeddings(self):
        initial_emb = torch.cat([self.user_emb, self.item_emb], dim=0)
        all_emb = [initial_emb]
        emb = initial_emb
        for _ in range(self.layers):
            emb = torch.sparse.mm(self.norm_adj_matrix, emb)
            all_emb.append(emb)
        final_emb = sum(all_emb) / (self.layers + 1)
        final_user_emb = final_emb[:self.num_users, :]
        final_item_emb = final_emb[self.num_users:, :]
        return final_user_emb, final_item_emb

    def bpr_loss(self, final_user_emb, final_item_emb, uids, pos, neg):
        user_e = final_user_emb[uids]
        pos_e = final_item_emb[pos]
        neg_e = final_item_emb[neg]
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        # loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss

    def reg_loss(self, uids, pos, neg):
        user_e = self.user_emb[uids]
        pos_e = self.item_emb[pos]
        neg_e = self.item_emb[neg]
        # 논문에서 reg loss 다르게 사용함.
        l2_reg = (1/2)*(user_e.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)) / user_e.shape[0]
        return self.reg_weight * l2_reg
    
    def forward(self, uids, pos, neg):
        final_user_emb, final_item_emb = self.get_embeddings()
        bpr_loss = self.bpr_loss(final_user_emb, final_item_emb, uids, pos, neg)
        reg_loss = self.reg_loss(uids, pos, neg)
        total_loss = bpr_loss + reg_loss
        return total_loss, bpr_loss, reg_loss

MODEL = LightGCN