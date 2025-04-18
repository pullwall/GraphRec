import torch
import numpy as np
import wandb
from data import RecDataset, shuffle, minibatch

def train_model(model, optimizer, device, dataset: RecDataset, epoch):
    S = dataset.UniformSample_original_python()
    uids = torch.Tensor(S[:, 0]).to(device)
    pos = torch.Tensor(S[:, 1]).to(device)
    neg = torch.Tensor(S[:, 2]).to(device)

    # Shuffle & minibatch
    uids, pos, neg = shuffle(uids, pos, neg)

    epoch_loss, epoch_bpr_loss, epoch_reg_loss, epoch_ssl_loss = 0, 0, 0, 0
    num_batches = 0

    for batch_uids, batch_pos, batch_neg in minibatch(uids, pos, neg):
        # batch마다 augmentation
        sub_adj1 = model.create_adj_mat(is_subgraph=True, aug_type=model.ssl_aug_type)
        sub_norm_adj1 = model.dataset.scipy_sparse_mat_to_torch_sparse_tensor(sub_adj1).to(device)

        sub_adj2 = model.create_adj_mat(is_subgraph=True, aug_type=model.ssl_aug_type)
        sub_norm_adj2 = model.dataset.scipy_sparse_mat_to_torch_sparse_tensor(sub_adj2).to(device)
        
        num_batches += 1
        optimizer.zero_grad()
        total_loss, bpr_loss, reg_loss, ssl_loss = model(
            batch_uids, batch_pos, batch_neg, sub_norm_adj1, sub_norm_adj2
        )
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.cpu().item()
        epoch_bpr_loss += bpr_loss.cpu().item()
        epoch_reg_loss += reg_loss.cpu().item()
        epoch_ssl_loss += ssl_loss.cpu().item()

    log_info = {
        "Total Loss": epoch_loss / num_batches,
        "BPR Loss": epoch_bpr_loss / num_batches,
        "Reg Loss": epoch_reg_loss / num_batches,
        "SSL Loss": epoch_ssl_loss / num_batches,
    }

    wandb.log({
        "Train/Total Loss": log_info["Total Loss"],
        "Train/BPR Loss": log_info["BPR Loss"],
        "Train/Reg Loss": log_info["Reg Loss"],
        "Train/SSL Loss": log_info["SSL Loss"],
    }, step=epoch)

    return log_info


def predict(user_emb, item_emb, uids_batch, train_csr, device):
    preds = user_emb[uids_batch] @ item_emb.t()
    # 훈련 시 사용한 아이템은 마스킹 (이미 interaction 한 아이템은 제외)
    mask = train_csr[uids_batch.cpu().numpy()].toarray()
    mask = torch.Tensor(mask).to(device)
    preds = preds * (1 - mask) - 1e8 * mask
    predictions = preds.argsort(descending=True, dim=1)
    return predictions.cpu().detach().numpy()

def calculate_metrics(uids, predictions, top_k, test_labels=None):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:top_k])
        label = test_labels[uid]
        if len(label) > 0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(top_k, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit += 1
                    loc = prediction.index(item)
                    dcg += np.reciprocal(np.log2(loc + 2))
            all_recall += hit / len(label)
            all_ndcg += (dcg / idcg if idcg > 0 else 0)
            user_num += 1
    return all_recall / user_num, all_ndcg / user_num

def evaluate_metrics(model, device, dataset, config):
    batch_user = config['batch_user']
    top_ks = config['top_k']
    max_k = max(top_ks)
    uids_all = np.arange(dataset.num_users)
    n_batch_user = int(np.ceil(len(uids_all) / batch_user))
    train_csr = dataset.train.tocsr()

    recall_dict = {k: 0.0 for k in top_ks}
    ndcg_dict = {k: 0.0 for k in top_ks}

    # 평가 시 full graph propagation 사용
    user_emb, item_emb = model.get_embeddings()

    for batch in range(n_batch_user):
        start = batch * batch_user
        end = min((batch + 1) * batch_user, len(uids_all))
        uids_batch = torch.LongTensor(uids_all[start:end]).to(device)

        predictions = predict(user_emb, item_emb, uids_batch, train_csr, device)
        predictions = predictions[:, :max_k]

        for k in top_ks:
            sliced_preds = predictions[:, :k]
            r, n = calculate_metrics(
                uids_batch.cpu().numpy(),
                sliced_preds,
                top_k=k,
                test_labels=dataset.test_labels
            )
            recall_dict[k] += r
            ndcg_dict[k] += n

    log_info = {
        **{f"Recall@{k}": np.round(recall_dict[k] / n_batch_user, 4) for k in top_ks},
        **{f"NDCG@{k}": np.round(ndcg_dict[k] / n_batch_user, 4) for k in top_ks},
    }

    return log_info
