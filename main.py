import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib
import wandb
from config import get_config
from utils import set_seed
from data import RecDataset
import time


torch.cuda.set_device(1)

def get_model(model_name, config, dataset):
    try:
        module = importlib.import_module(f"models.{model_name}")
        cls = module.MODEL
    except ModuleNotFoundError:
        raise ValueError(f"Model module 'models.{model_name}' not found.")
    except AttributeError:
        raise ValueError(f"Module 'models.{model_name}' does not define MODEL.")
    return cls(config, dataset)

def get_trainer(model_name):
    try:
        module = importlib.import_module(f"trainers.{model_name}_trainer")
        train_model = module.train_model
        evaluate_metrics = module.evaluate_metrics
    except ModuleNotFoundError:
        raise ValueError(f"Trainer module 'trainers.{model_name}_trainer' not found.")
    except AttributeError as e:
        raise ValueError(f"Module 'trainers.{model_name}_trainer' must define 'train_model' and 'evaluate_metrics': {e}")
    return train_model, evaluate_metrics

def main():
    config = get_config()
    set_seed(config['seed'])

    wandb.init(
        project=f"{config['model']}",
        config=config,
        name=f"{config['model']}_{config['dataset']}_reg{config['ssl_reg']}_temp{config['ssl_temp']}_ratio{config['ssl_ratio']}",
    )

    dataset = RecDataset(config['dataset'])
    model = get_model(config['model'], config, dataset)
    device = dataset.device
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    train_model, evaluate_metrics = get_trainer(config['model'])

    # 초기 평가
    eval_log_info = evaluate_metrics(model, device, dataset, config)
    print("Initial evaluation:", eval_log_info)
    best_epoch = 0
    best_recall = eval_log_info.get('Recall@20', 0.0)
    best_ndcg = eval_log_info.get('NDCG@20', 0.0)

    for epoch in tqdm(range(1, config['epochs'] + 1)):
        train_start= time.time()
        train_log_info = train_model(model, optimizer, device, dataset, epoch)
        train_end = time.time()
        print(f"Epoch {epoch}: {train_log_info}")
        print(f"Train took {train_end-train_start:.2f} seconds")
        

        if epoch % 1 == 0:
            eval_start = time.time()
            eval_log_info = evaluate_metrics(model, device, dataset, config)
            eval_end = time.time()
            print(f"Epoch {epoch}: {eval_log_info}")
            print(f"Evaluation took {eval_end - eval_start:.2f} seconds")

            log_start = time.time()
            wandb_log_dict = {
                **{f"Eval/Recall@{k}": eval_log_info[f"Recall@{k}"] for k in config['top_k']},
                **{f"Eval/NDCG@{k}": eval_log_info[f"NDCG@{k}"] for k in config['top_k']}
            }
            wandb.log(wandb_log_dict, step=epoch)

            if eval_log_info['Recall@20'] >= best_recall:
                best_recall = eval_log_info['Recall@20']
                best_ndcg = eval_log_info['NDCG@20']
                best_epoch = epoch

                wandb_best_log = {
                    "Best/Recall@20": best_recall,
                    "Best/NDCG@20": best_ndcg,
                    "Best/Epoch": best_epoch
                }
                wandb.log(wandb_best_log, step=epoch)

                wandb_snapshot_log = {
                    **{f"Snapshot@Recall20/Recall@{k}": eval_log_info[f"Recall@{k}"] for k in config['top_k']},
                    **{f"Snapshot@Recall20/NDCG@{k}": eval_log_info[f"NDCG@{k}"] for k in config['top_k']},
                    "Snapshot@Recall20/BestEpoch": epoch
                }
                wandb.log(wandb_snapshot_log, step=epoch)
            log_end = time.time()
            print(f"Logging took {log_end - log_start:.2f} seconds")


    print("Best result (Recall@20 기준):")
    print(f"Epoch {best_epoch}: Recall@20 = {best_recall}, NDCG@20 = {best_ndcg}")


if __name__ == '__main__':
    main()
    

