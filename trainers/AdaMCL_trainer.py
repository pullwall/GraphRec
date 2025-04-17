import torch
import numpy as np
import wandb
from data import RecDataset, shuffle, minibatch

def train_model(model, optimizer, device, dataset:RecDataset, epoch):
    S = dataset.UniformSample_original_python()
    uids = torch.from_numpy(S[:, 0]).long().to(device)
    pos = torch.from_numpy(S[:, 1]).long().to(device)
    neg = torch.from_numpy(S[:, 2]).long().to(device)

    uids, pos, neg = shuffle(uids, pos, neg)

    epoch_loss, epoch_bpr_loss, epoch_reg_loss = 0, 0, 0
    epoch_ssl_loss, epoch_layerl_loss = 0, 0
    num_batches = 0

    for batch_uids, batch_pos, batch_neg in minibatch(uids, pos, neg, batch_size=dataset.batch_size):
        num_batches += 1
        optimizer.zero_grad()
        total_loss, bpr_loss, reg_loss, ssl_loss, layerl_loss = model(batch_uids, batch_pos, batch_neg)
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_bpr_loss += bpr_loss.item()
        epoch_reg_loss += reg_loss.item()
        epoch_ssl_loss += ssl_loss.item()
        epoch_layerl_loss += layerl_loss.item()

    log_info = {
        "Total Loss": epoch_loss / num_batches,
        "BPR Loss": epoch_bpr_loss / num_batches,
        "Reg Loss": epoch_reg_loss / num_batches,
        "SSL Loss": epoch_ssl_loss / num_batches,
        "Layer Loss": epoch_layerl_loss / num_batches
    }
    wandb.log({
        "Train/Total Loss": log_info["Total Loss"],
        "Train/BPR Loss": log_info["BPR Loss"],
        "Train/Reg Loss": log_info["Reg Loss"],
        "Train/SSL Loss": log_info["SSL Loss"],
        "Train/Layer Loss": log_info["Layer Loss"]
    }, step=epoch)

    return log_info


def predict(user_emb, item_emb, uids_batch, train_csr, device):
    preds = user_emb[uids_batch] @ item_emb.t()
    mask = train_csr[uids_batch.cpu().numpy()].toarray()
    mask = torch.Tensor(mask).to(device)
    preds = preds * (1 - mask) - 1e8 * mask
    return preds.argsort(descending=True, dim=1).cpu().numpy()

def calculate_metrics(uids, predictions, top_k=20, test_labels=None):
    all_recall, all_ndcg, user_num = 0, 0, 0
    for i in range(len(uids)):
        uid = uids[i]
        pred = list(predictions[i][:top_k])
        label = test_labels[uid]
        if label:
            hit = len(set(pred) & set(label))
            dcg = sum([1 / np.log2(pred.index(i) + 2) for i in label if i in pred])
            idcg = sum([1 / np.log2(i + 2) for i in range(min(len(label), top_k))])
            all_recall += hit / len(label)
            all_ndcg += dcg / idcg if idcg > 0 else 0
            user_num += 1
    return all_recall / user_num, all_ndcg / user_num

def evaluate_metrics(model, device, dataset, config):
    batch_user = config['batch_user']
    uids_all = np.arange(dataset.num_users)
    n_batch = int(np.ceil(len(uids_all) / batch_user))
    recall, ndcg = 0, 0
    train_csr = dataset.train.tocsr()
    with torch.no_grad():
        embs = model.get_embeddings()
        user_emb = embs['fused'][:dataset.num_users]
        item_emb = embs['fused'][dataset.num_users:]

        for b in range(n_batch):
            start, end = b * batch_user, min((b + 1) * batch_user, len(uids_all))
            uids_batch = torch.LongTensor(uids_all[start:end]).to(device)
            preds = predict(user_emb, item_emb, uids_batch, train_csr, device)
            r, n = calculate_metrics(uids_batch.cpu().numpy(), preds, 20, dataset.test_labels)
            recall += r
            ndcg += n
    return {
        "Recall@20": round(recall / n_batch, 4),
        "NDCG@20": round(ndcg / n_batch, 4)
    }
