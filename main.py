import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import importlib
import wandb
from config import get_config
from utils import set_seed
from data import RecDataset


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

    eval_log_info = evaluate_metrics(model, device, dataset, config)
    best_epoch, best_recall, best_ndcg = 0, eval_log_info['Recall@20'], eval_log_info['NDCG@20']
    print("Initial evaluation:", eval_log_info)
    
    # Train
    for epoch in tqdm(range(1, config['epochs'] + 1)):
        train_log_info = train_model(model, optimizer, device, dataset, epoch)
        print(f"Epoch {epoch}: {train_log_info}")
        if epoch % 5 == 0:
            eval_log_info = evaluate_metrics(model, device, dataset, config)
            print(f"Epoch {epoch}: {eval_log_info}")
            wandb.log({
                "Eval/Recall@20": eval_log_info["Recall@20"],
                "Eval/NDCG@20": eval_log_info["NDCG@20"],
            },step=epoch)
            
            if eval_log_info['Recall@20'] >= best_recall:
                best_recall = eval_log_info['Recall@20']
                best_ndcg = eval_log_info['NDCG@20']
                best_epoch = epoch
    
    best_result = {"Best Epoch": best_epoch, "Best Recall@20": best_recall, "Best NDCG@20": best_ndcg}
    print("Best result:", best_result)

if __name__ == '__main__':
    main()
    

# trial#1 Best result: {'Best Epoch': 820, 'Best Recall@20': 0.1787, 'Best NDCG@20': 0.152} 
# -> xavier initialization, python neg_sampling
# trial#2 Best result: {'Best Epoch': 720, 'Best Recall@20': 0.1544, 'Best NDCG@20': 0.1301}
# -> uniform initialization, cpp neg_sampling
