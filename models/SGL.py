import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BPR import BPR
                        
class SGL(BPR):
    def __init__(self, config, dataset):
        super(SGL, self).__init__(config, dataset)
        self.layers = config['layers']
        self.adj_matrix = dataset.adj_matrix
        self.norm_adj_matrix = dataset.norm_adj_matrix
        # SSL 관련 하이퍼파라미터 추가
        self.ssl_reg = config.get('ssl_reg', 0.1)
        self.ssl_temp = config.get('ssl_temp', 0.5)
        self.ssl_aug_type = config.get('aug_type', 'ed')
        self.ssl_ratio = config.get('ssl_ratio', 0.2)

    def init_embedding(self, num, dim):
        emb = torch.empty(num, dim, device=self.dataset.device)
        # 논문에서 권장하는 normal(std=0.1) 초기화
        nn.init.normal_(emb, std=0.1)
        return emb

    def get_embeddings(self):
        # full graph propagation (supervised branch)
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

    def get_embeddings_aug(self, norm_adj):
        # propagation using augmented normalized adjacency matrix
        initial_emb = torch.cat([self.user_emb, self.item_emb], dim=0)
        all_emb = [initial_emb]
        emb = initial_emb
        for _ in range(self.layers):
            emb = torch.sparse.mm(norm_adj, emb)
            all_emb.append(emb)
        final_emb = sum(all_emb) / (self.layers + 1)
        final_user_emb = final_emb[:self.num_users, :]
        final_item_emb = final_emb[self.num_users:, :]
        return final_user_emb, final_item_emb

    def bpr_loss(self, full_user_emb, full_item_emb, uids, pos, neg):
        user_e = full_user_emb[uids]
        pos_e = full_item_emb[pos]
        neg_e = full_item_emb[neg]
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        # softplus 손실 사용
        loss = torch.mean(F.softplus(neg_scores - pos_scores))
        return loss

    def reg_loss(self, uids, pos, neg):
        user_e = self.user_emb[uids]
        pos_e = self.item_emb[pos]
        neg_e = self.item_emb[neg]
        l2_reg = (user_e.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)) / (2 * user_e.shape[0])
        return self.reg_weight * l2_reg
    
    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        import numpy as np
        import scipy.sparse as sp

        def randint_choice(n, size=None, replace=False):
            return np.random.choice(n, size=size, replace=replace)

        n_nodes = self.num_users + self.num_items
        train_coo = self.dataset.train.tocoo()
        users_np, items_np = train_coo.row, train_coo.col


        if is_subgraph and self.ssl_ratio > 0:
            if aug_type == 'nd':
                # 노드 드랍: 각 사용자와 아이템에서 ssl_ratio 만큼의 인덱스를 드랍
                drop_user_idx = randint_choice(self.num_users, size=int(self.num_users * self.ssl_ratio), replace=False)
                drop_item_idx = randint_choice(self.num_items, size=int(self.num_items * self.ssl_ratio), replace=False)
                indicator_user = np.ones(self.num_users, dtype=np.float32)
                indicator_item = np.ones(self.num_items, dtype=np.float32)
                indicator_user[drop_user_idx] = 0.0
                indicator_item[drop_item_idx] = 0.0
                diag_indicator_user = sp.diags(indicator_user)
                diag_indicator_item = sp.diags(indicator_item)
                # 원본 사용자-아이템 행렬 (shape: [num_users, num_items])
                R = sp.csr_matrix((np.ones_like(users_np, dtype=np.float32), (users_np, items_np)),
                                shape=(self.num_users, self.num_items))
                # 드랍된 노드를 반영한 행렬
                R_prime = diag_indicator_user.dot(R).dot(diag_indicator_item)
                user_np_keep, item_np_keep = R_prime.nonzero()
                ratings_keep = R_prime.data
                tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + self.num_users)),
                                        shape=(n_nodes, n_nodes))
            elif aug_type in ['ed', 'rw']:
                # 엣지 드랍: 전체 엣지 중 ssl_ratio 만큼 드랍하고 나머지를 유지
                total_edges = len(users_np)
                keep_num = int(total_edges * (1 - self.ssl_ratio))
                keep_idx = randint_choice(total_edges, size=keep_num, replace=False)
                user_np = np.array(users_np)[keep_idx]
                item_np = np.array(items_np)[keep_idx]
                ratings = np.ones_like(user_np, dtype=np.float32)
                tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)),
                                        shape=(n_nodes, n_nodes))
        else:
            ratings = np.ones_like(users_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (users_np, items_np + self.num_users)),
                                    shape=(n_nodes, n_nodes))
        
        # 대칭 인접행렬 구성: 사용자-아이템 + 아이템-사용자
        adj_mat = tmp_adj + tmp_adj.T

        # 정규화: D^{-1/2} A D^{-1/2}
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)

        return norm_adj


    def forward(self, uids, pos, neg, sub_norm_adj1, sub_norm_adj2):
        # supervised branch: full graph propagation
        full_user_emb, full_item_emb = self.get_embeddings()
        bpr_loss = self.bpr_loss(full_user_emb, full_item_emb, uids, pos, neg)
        reg_loss = self.reg_loss(uids, pos, neg)
        
        # self-supervised branch: propagation on augmented graphs
        user_emb1, item_emb1 = self.get_embeddings_aug(sub_norm_adj1)
        user_emb2, item_emb2 = self.get_embeddings_aug(sub_norm_adj2)
        # 정규화: cosine similarity 기반 SSL 손실을 위해
        user_emb1 = F.normalize(user_emb1, dim=1)
        item_emb1 = F.normalize(item_emb1, dim=1)
        user_emb2 = F.normalize(user_emb2, dim=1)
        item_emb2 = F.normalize(item_emb2, dim=1)
        
        # --- SSL 손실 계산 (User & Item 각각 InfoNCE 방식) ---
        # 사용자 SSL 손실: 배치 내 사용자들에 대해, 두 augmented embedding 간의 유사도를 모두 비교
        u_emb1_batch = user_emb1[uids]                     # (batch_size x dim)
        pos_logits_user = torch.sum(u_emb1_batch * user_emb2[uids], dim=1, keepdim=True)
        tot_logits_user = torch.matmul(u_emb1_batch, user_emb2.t())  # (batch_size x num_users)
        ssl_logits_user = tot_logits_user - pos_logits_user
        
        # 아이템 SSL 손실: 양의 아이템에 대해 동일하게
        pos_emb1_batch = item_emb1[pos]
        pos_logits_item = torch.sum(pos_emb1_batch * item_emb2[pos], dim=1, keepdim=True)
        tot_logits_item = torch.matmul(pos_emb1_batch, item_emb2.t())  # (batch_size x num_items)
        ssl_logits_item = tot_logits_item - pos_logits_item
        
        # temperature scaling 후 logsumexp (InfoNCE 스타일)
        clogits_user = torch.logsumexp(ssl_logits_user / self.ssl_temp, dim=1)
        clogits_item = torch.logsumexp(ssl_logits_item / self.ssl_temp, dim=1)
        ssl_loss = torch.mean(clogits_user + clogits_item)
        
        total_loss = bpr_loss + reg_loss + self.ssl_reg * ssl_loss
        return total_loss, bpr_loss, reg_loss, ssl_loss

MODEL = SGL
