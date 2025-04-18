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

    epoch_loss, epoch_bpr_loss, epoch_reg_loss, epoch_cl_loss = 0, 0, 0, 0
    num_batches = 0

    for batch_uids, batch_pos, batch_neg in minibatch(uids, pos, neg):
        num_batches += 1
        optimizer.zero_grad()
        total_loss, bpr_loss, reg_loss, cl_loss = model(batch_uids, batch_pos, batch_neg)
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_bpr_loss += bpr_loss.item()
        epoch_reg_loss += reg_loss.item()
        epoch_cl_loss += cl_loss.item()

    log_info = {
        "Total Loss": epoch_loss / num_batches,
        "BPR Loss": epoch_bpr_loss / num_batches,
        "Reg Loss": epoch_reg_loss / num_batches,
        "CL Loss": epoch_cl_loss / num_batches,
    }

    wandb.log({
        "Train/Total Loss": log_info["Total Loss"],
        "Train/BPR Loss": log_info["BPR Loss"],
        "Train/Reg Loss": log_info["Reg Loss"],
        "Train/CL Loss": log_info["CL Loss"],
    }, step=epoch)

    return log_info

def predict(user_emb, item_emb, uids_batch, train_csr, device):
    preds = user_emb[uids_batch] @ item_emb.t()
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

    recall_dict = {k: 0.0 for k in top_ks}
    ndcg_dict = {k: 0.0 for k in top_ks}

    train_csr = dataset.train.tocsr()
    user_emb, item_emb = model.get_embeddings()  # dropout 없이 추출

    for batch in range(n_batch_user):
        start = batch * batch_user
        end = min((batch + 1) * batch_user, len(uids_all))
        uids_batch = torch.LongTensor(uids_all[start:end]).to(device)

        # max_k만큼 top 추천 예측
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
        **{f"NDCG@{k}": np.round(ndcg_dict[k] / n_batch_user, 4) for k in top_ks}
    }

    return log_info
