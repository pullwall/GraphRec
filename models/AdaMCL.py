import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import math
import os
from tqdm import tqdm
from scipy.sparse import hstack, vstack, csr_matrix
from models.BPR import BPR
from config import get_config


class AdaMCL(BPR):
    def __init__(self, config, dataset):
        super(AdaMCL, self).__init__(config, dataset)
        # graph reg weight: lambda_1
        # layer reg weight: lambda_2
        # reg loss weight: lambda_3
        # alpha: g + alpha*ghat
        self.layers = config['layers']
        self.hyper_layer = config.get('hyper_layer', math.ceil(self.layers / 2))
        self.gamma = config.get('gamma', 1)
        self.ssl_temp = config['ssl_temp']
        self.ssl_reg = config['ssl_reg']
        self.layer_reg = config.get('layer_reg', 0.1)
        self.alpha = config.get('alpha', 0.2)
        self.threshold = config.get('threshold')
        self.topk = config.get('topk')

        self.norm_adj = dataset.norm_adj_matrix
        self.my_adj = self.get_my_adj_matrix(dataset.train).to(self.device)

        # make d_i
        deg = np.array(dataset.adj_matrix.sum(axis=1)).flatten()
        deg[deg == 0] = 1
        d_i = np.log(deg) / np.mean(np.log(deg))
        self.d_i = torch.tensor(d_i).reshape(-1, 1).float().to(self.device)


    def get_my_adj_matrix(self, inter_matrix):
        save_path = f"./data/{self.config['dataset']}/{self.config['dataset']}_my_adj.npz"
        if os.path.exists(save_path):
            print(f"Loading saved MyAdj matrix from {save_path}")
            mat = sp.load_npz(save_path)
        else:
            print(f"Computing MyAdj matrix ...")
            R = inter_matrix.tocsr()
            num_users, num_items = self.num_users, self.num_items

            def build_adj(R, threshold, topk):
                sim = R @ R.T  # user-user or item-item similarity
                sim = sim.tocoo()

                row_sum = np.array(R.sum(axis=1)).flatten()
                u_sum = row_sum[sim.row]
                v_sum = row_sum[sim.col]
                inter = sim.data
                union = u_sum + v_sum - inter
                jaccard = inter / (union + 1e-10)

                mask = (sim.row != sim.col) & (jaccard > threshold)
                jaccard = jaccard[mask]
                rows = sim.row[mask]
                cols = sim.col[mask]

                dim = R.shape[0]
                sim_matrix = sp.coo_matrix((jaccard, (rows, cols)), shape=(dim, dim)).tocsr()

                data, row, col = [], [], []
                for i in tqdm(range(dim), desc="Top-k filtering"):
                    row_i = sim_matrix[i].toarray().flatten()
                    if np.count_nonzero(row_i) == 0:
                        continue
                    topk_idx = np.argsort(-row_i)[:topk]
                    topk_val = row_i[topk_idx]
                    norm_val = topk_val / (np.sum(topk_val) + 1e-10)

                    row.extend([i] * len(topk_idx))
                    col.extend(topk_idx)
                    data.extend(norm_val)

                return sp.coo_matrix((data, (row, col)), shape=(dim, dim))

            U_U = build_adj(R, threshold=self.threshold, topk=self.topk)
            I_I = build_adj(R.T, threshold=self.threshold, topk=self.topk)

            print("Making U_U, I_I complete!")
            
            zero_ui = csr_matrix((U_U.shape[0], I_I.shape[1]))  # upper right
            zero_iu = csr_matrix((I_I.shape[0], U_U.shape[1]))  # lower left

            # 상단: [U_U | 0]
            upper = hstack([U_U, zero_ui], format='csr')

            # 하단: [0   | I_I]
            lower = hstack([zero_iu, I_I], format='csr')

            mat = vstack([upper, lower], format='csr')

            sp.save_npz(save_path, mat)
            print(f"MyAdj matrix saved to {save_path}")

        mat = mat.tocoo()
        indices = torch.LongTensor([mat.row, mat.col])
        values = torch.FloatTensor(mat.data)
        return torch.sparse.FloatTensor(indices, values, torch.Size(mat.shape)).to(self.device)


    def get_embeddings(self):
        ego = torch.cat([self.user_emb, self.item_emb], dim=0)
        embeddings = [ego]
        interaction_list, neighbor_list = [], []

        for layer in range(1, self.layers + 1):
            inter = torch.sparse.mm(self.norm_adj, ego)
            collab = torch.sparse.mm(self.my_adj, ego)

            interaction_list.append(inter)
            neighbor_list.append(collab)

            # inter + collab 한게 저자 코드. 논문대로라면 collab만 사용해야 함.
            sim = F.normalize(inter, dim=1) * F.normalize(inter + collab, dim=1)
            sim = sim.sum(dim=1, keepdim=True).clamp(min=0.0)
            beta = self.gamma / (layer + sim * self.d_i)
            
            fused = inter + collab * beta
            embeddings.append(fused)
            ego = fused

        emb_stack = torch.stack(embeddings, dim=1)
        fused_emb = emb_stack.mean(dim=1)

        interaction_emb = torch.stack(interaction_list, dim=1).mean(dim=1)
        neighbor_emb = torch.stack(neighbor_list, dim=1).mean(dim=1)
        mid_emb = embeddings[self.hyper_layer]

        return {
            'fused': fused_emb,
            'interaction': interaction_emb,
            'neighbor': neighbor_emb,
            'mid': mid_emb
        }

    # Fused view 용으로 override
    def bpr_loss(self, user_fused, item_fused, uids, pos, neg):
        u_emb = user_fused[uids]
        pos_emb = item_fused[pos]
        neg_emb = item_fused[neg]

        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        return loss
    
    
    def contrastive_loss(self, a, b, temp, reg):
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        pos_score = torch.sum(a * b, dim=1)
        ttl_score = torch.matmul(a, b.T)
        pos_score = torch.exp(pos_score / temp)
        ttl_score = torch.exp(ttl_score / temp).sum(dim=1)
        loss = -torch.log(pos_score / ttl_score + 1e-10).mean()
        return reg * loss


    def forward(self, uids, pos, neg):
        embs = self.get_embeddings()
        user_fused = embs['fused'][:self.num_users]
        item_fused = embs['fused'][self.num_users:]

        # BPR loss
        bpr_loss = self.bpr_loss(user_fused, item_fused, uids, pos, neg)

        # Regularization
        reg_loss = self.reg_loss(uids, pos, neg)

        # Multi-view CL (fused vs interaction / neighbor)
        u_fused, u_inter, u_neigh = user_fused[uids], embs['interaction'][:self.num_users][uids], embs['neighbor'][:self.num_users][uids]
        i_fused, i_inter, i_neigh = item_fused[pos], embs['interaction'][self.num_users:][pos], embs['neighbor'][self.num_users:][pos]

        # Graph loss
        ssl_loss = self.contrastive_loss(u_fused, u_inter, self.ssl_temp, self.ssl_reg)
        ssl_loss += self.contrastive_loss(i_fused, i_inter, self.ssl_temp, self.ssl_reg)
        ssl_loss += self.contrastive_loss(u_fused, u_neigh, self.ssl_temp, self.alpha * self.ssl_reg)
        ssl_loss += self.contrastive_loss(i_fused, i_neigh, self.ssl_temp, self.alpha * self.ssl_reg)

        # Layer-level contrast (fused vs mid-layer embedding)
        u_mid = embs['mid'][:self.num_users][uids]
        i_mid = embs['mid'][self.num_users:][pos]
        layer_loss = self.contrastive_loss(u_fused, u_mid, self.ssl_temp, self.layer_reg)
        layer_loss += self.contrastive_loss(i_fused, i_mid, self.ssl_temp, self.layer_reg)
        total_loss = bpr_loss + reg_loss + ssl_loss + layer_loss
        
        return total_loss, bpr_loss, reg_loss, ssl_loss, layer_loss

MODEL = AdaMCL
