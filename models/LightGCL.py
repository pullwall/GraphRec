import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BPR import BPR
from utils import sparse_dropout

class LightGCL(BPR):
    def __init__(self, config, dataset):
        super(LightGCL, self).__init__(config, dataset)
        self.layers = config.get('layers')
        self.dropout = config.get('ssl_ratio')
        self.temp = config.get('ssl_temp')
        self.lambda_1 = config.get('ssl_reg')
        self.lambda_2 = config.get('reg_weight')
        self.svd_q = config.get('svd_q')
        self.norm_adj_matrix = dataset.norm_adj_matrix
        self.compute_svd_matrices()

    def compute_svd_matrices(self):
        adj = self.norm_adj_matrix.coalesce()
        svd_u, s, svd_v = torch.svd_lowrank(adj, q=self.svd_q)
        self.u_mul_s = svd_u @ torch.diag(s)
        self.v_mul_s = svd_v @ torch.diag(s)
        self.ut = svd_u.t()
        self.vt = svd_v.t()
        del s

    def bpr_loss(self, user_emb, item_emb, uids, pos, neg):
        u_emb = user_emb[uids]
        pos_emb = item_emb[pos]
        neg_emb = item_emb[neg]
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        return loss

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
    
    def get_embeddings_svd(self):
        # 초기 임베딩: 사용자와 아이템 임베딩을 따로 사용
        E_u = self.user_emb      # shape: [num_users, d]
        E_i = self.item_emb      # shape: [num_items, d]
        E_u_list = [E_u]
        E_i_list = [E_i]
        G_u_list = [E_u]
        G_i_list = [E_i]

        n_users = self.num_users
        n_items = self.num_items

        # 전체 정방 인접행렬에서 인덱스와 값을 가져옴
        adj = self.norm_adj_matrix.coalesce()
        indices = adj.indices()
        values = adj.values()

        # 사용자 → 아이템 블록: 행 인덱스 < n_users, 열 인덱스 >= n_users
        mask_ui = (indices[0] < n_users) & (indices[1] >= n_users)
        rows_ui = indices[0][mask_ui]
        cols_ui = indices[1][mask_ui] - n_users  # 아이템 인덱스 로컬화
        values_ui = values[mask_ui]
        adj_ui = torch.sparse_coo_tensor(
            torch.stack([rows_ui, cols_ui]),
            values_ui,
            size=(n_users, n_items)
        ).coalesce().to(self.device)

        # 아이템 → 사용자 블록: 행 인덱스 >= n_users, 열 인덱스 < n_users
        mask_iu = (indices[0] >= n_users) & (indices[1] < n_users)
        rows_iu = indices[0][mask_iu] - n_users  # 아이템 인덱스 로컬화
        cols_iu = indices[1][mask_iu]
        values_iu = values[mask_iu]
        adj_iu = torch.sparse_coo_tensor(
            torch.stack([rows_iu, cols_iu]),
            values_iu,
            size=(n_items, n_users)
        ).coalesce().to(self.device)

        for _ in range(self.layers):
            # 각 블록에 대해 dropout 적용
            dropped_adj_ui = sparse_dropout(adj_ui, self.dropout)
            dropped_adj_iu = sparse_dropout(adj_iu, self.dropout)

            # GNN propagation: 사용자 임베딩 업데이트는 아이템 임베딩을 통해, 아이템 임베딩 업데이트는 사용자 임베딩을 통해
            Z_u = torch.sparse.mm(dropped_adj_ui, E_i)   # [n_users, d]
            Z_i = torch.sparse.mm(dropped_adj_iu, E_u)   # [n_items, d]

            # SVD propagation: SVD 행렬은 전체 노드에 대해 계산되었으므로, 각 블록에 해당하는 부분만 사용
            G_u = self.u_mul_s[:n_users, :] @ (self.vt[:, n_users:] @ E_i)   # [n_users, d]
            G_i = self.v_mul_s[n_users:, :] @ (self.ut[:, :n_users] @ E_u)   # [n_items, d]

            # 다음 레이어 입력은 GNN 전파 결과
            E_u = Z_u
            E_i = Z_i

            E_u_list.append(E_u)
            E_i_list.append(E_i)
            G_u_list.append(G_u)
            G_i_list.append(G_i)

        final_E_u = sum(E_u_list)
        final_E_i = sum(E_i_list)
        final_G_u = sum(G_u_list)
        final_G_i = sum(G_i_list)
        return final_E_u, final_E_i, final_G_u, final_G_i



    def forward(self, uids, pos, neg):
        # 1. BPR용 임베딩 (dropout X)
        user_emb, item_emb = self.get_embeddings()
        loss_bpr = self.bpr_loss(user_emb, item_emb, uids, pos, neg)
        loss_reg = self.reg_loss(uids, pos, neg)

        # 2. Contrastive용 임베딩 (dropout + SVD 포함)
        E_u, E_i, G_u, G_i = self.get_embeddings_svd()

        # contrastive loss
        neg_score_user = torch.log(torch.exp(G_u[uids] @ E_u.t() / self.temp).sum(1) + 1e-8).mean()
        neg_score_item = torch.log(torch.exp(G_i[pos] @ E_i.t() / self.temp).sum(1) + 1e-8).mean()
        pos_score_user = (torch.clamp((G_u[uids] * E_u[uids]).sum(1) / self.temp, -5.0, 5.0)).mean()
        pos_score_item = (torch.clamp((G_i[pos] * E_i[pos]).sum(1) / self.temp, -5.0, 5.0)).mean()
        loss_ssl = - (pos_score_user + pos_score_item) + (neg_score_user + neg_score_item)

        total_loss = loss_bpr + loss_reg + self.lambda_1 * loss_ssl
        return total_loss, loss_bpr, loss_reg, loss_ssl


MODEL = LightGCL