import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BPR import BPR

class SimGCL(BPR):
    def __init__(self, config, dataset):
        super(SimGCL, self).__init__(config, dataset)
        self.layers = config['layers']
        self.ssl_reg = config.get('ssl_reg', 0.1)
        self.eps = config.get('eps', 0.1)
        self.ssl_temp = config.get('ssl_temp', 0.2)  # 0.2 고정 권장
        self.norm_adj_matrix = dataset.norm_adj_matrix

    def init_embedding(self, num, dim):
        emb = torch.empty(num, dim, device=self.dataset.device)
        # 논문대로 xavier 초기화
        nn.init.xavier_uniform_(emb)
        return emb
    

    def get_embeddings(self):
        # 기본 LightGCN propagation (supervised branch)
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

    def get_embeddings_perturbed(self):
        # perturbed propagation: 각 레이어마다 작은 노이즈(perturbation)를 추가
        initial_emb = torch.cat([self.user_emb, self.item_emb], dim=0)
        all_emb = []
        emb = initial_emb
        for _ in range(self.layers):
            emb = torch.sparse.mm(self.norm_adj_matrix, emb)
            noise = torch.rand_like(emb)
            noise = F.normalize(noise, p=2, dim=1)
            emb = emb + torch.sign(emb) * noise * self.eps
            all_emb.append(emb)
        final_emb = sum(all_emb) / self.layers
        final_user_emb = final_emb[:self.num_users, :]
        final_item_emb = final_emb[self.num_users:, :]
        return final_user_emb, final_item_emb

    def bpr_loss(self, full_user_emb, full_item_emb, uids, pos, neg):
        user_e = full_user_emb[uids]
        pos_e = full_item_emb[pos]
        neg_e = full_item_emb[neg]
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss

    def reg_loss(self, uids, pos, neg):
        user_e = self.user_emb[uids]
        pos_e = self.item_emb[pos]
        neg_e = self.item_emb[neg]
        l2_reg = (user_e.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)) / (2 * user_e.shape[0])
        return self.reg_weight * l2_reg

    def forward(self, uids, pos, neg):
        full_user_emb, full_item_emb = self.get_embeddings()
        loss_bpr = self.bpr_loss(full_user_emb, full_item_emb, uids, pos, neg)
        loss_reg = self.reg_loss(uids, pos, neg)
        
        # contrastive branch: 두 번의 perturbed propagation을 통해 노이즈가 추가된 임베딩을 생성
        perturbed_user_emb1, perturbed_item_emb1 = self.get_embeddings_perturbed()
        perturbed_user_emb2, perturbed_item_emb2 = self.get_embeddings_perturbed()

        perturbed_user_emb1 = F.normalize(perturbed_user_emb1, dim=1)
        perturbed_item_emb1 = F.normalize(perturbed_item_emb1, dim=1)
        perturbed_user_emb2 = F.normalize(perturbed_user_emb2, dim=1)
        perturbed_item_emb2 = F.normalize(perturbed_item_emb2, dim=1)
        
        uids = torch.unique(uids)
        pos = torch.unique(pos)
        
        # 사용자에 대한 contrastive loss
        u_emb1_batch = perturbed_user_emb1[uids]  # (batch_size x dim)
        u_emb2_batch = perturbed_user_emb2[uids]
        pos_logits_user = torch.sum(u_emb1_batch * u_emb2_batch, dim=1, keepdim=True)
        tot_logits_user = torch.matmul(u_emb1_batch, perturbed_user_emb2.t())
        ssl_logits_user = tot_logits_user - pos_logits_user
        clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
        
        # 아이템에 대한 contrastive loss (positive sample로 사용한 아이템)
        pos_emb1_batch = perturbed_item_emb1[pos]
        pos_emb2_batch = perturbed_item_emb2[pos]
        pos_logits_item = torch.sum(pos_emb1_batch * pos_emb2_batch, dim=1, keepdim=True)
        tot_logits_item = torch.matmul(pos_emb1_batch, perturbed_item_emb2.t())
        ssl_logits_item = tot_logits_item - pos_logits_item
        clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
        
        loss_ssl = torch.mean(clogits_user + clogits_item)
        
        total_loss = loss_bpr + loss_reg + self.ssl_reg * loss_ssl
        return total_loss, loss_bpr, loss_reg, loss_ssl

MODEL = SimGCL