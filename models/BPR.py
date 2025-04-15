import torch
import torch.nn as nn

class BPR(nn.Module):
    def __init__(self, config, dataset):
        super(BPR, self).__init__()
        self.dataset = dataset
        self.device = dataset.device
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.embedding_dim = config['embedding_dim']
        self.reg_weight = config['reg_weight']
        self.user_emb = nn.Parameter(self.init_embedding(self.num_users, self.embedding_dim))
        self.item_emb = nn.Parameter(self.init_embedding(self.num_items, self.embedding_dim))

    def init_embedding(self, num, dim):
        emb = torch.empty(num, dim, device=self.dataset.device)
        nn.init.xavier_uniform_(emb)
        return emb

    def get_embeddings(self):
        return self.user_emb, self.item_emb

    def forward(self, uids, pos, neg):
        bpr_loss = self.bpr_loss(uids, pos, neg)
        reg_loss = self.reg_loss(uids, pos, neg)
        total_loss = bpr_loss + reg_loss
        return total_loss, bpr_loss, reg_loss

    def bpr_loss(self, uids, pos, neg):
        user_e = self.user_emb[uids]
        pos_e = self.item_emb[pos]
        neg_e = self.item_emb[neg]
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        return loss

    def reg_loss(self, uids, pos, neg):
        user_e = self.user_emb[uids]
        pos_e = self.item_emb[pos]
        neg_e = self.item_emb[neg]
        l2_reg = (user_e.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)) / user_e.shape[0]
        return self.reg_weight * l2_reg
    
MODEL = BPR