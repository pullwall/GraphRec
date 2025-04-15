import argparse

# Hyperparameter
# cl loss -> {0.01, 0.05, 0.1, 0.2, 0.5, 1.0}

DEFAULTS = {
    'bpr': {
         'epochs': 1000,
         'lr': 1e-3,
         'batch_size': 2048,
         'batch_user': 256,
         'seed': 2025,
         'embedding_dim': 64,
         'layers': 3,
         'reg_weight': 1e-4,
    },
    'LightGCN': {
         'epochs': 1000,
         'lr': 1e-3,
         'batch_size': 2048,
         'batch_user': 256,
         'seed': 2025,
         'embedding_dim': 64,
         'layers': 4,
         'reg_weight': 1e-4,
    },
    'SGL': {
         'epochs': 50,
         'lr': 1e-3,
         'batch_size': 2048,
         'batch_user': 256,
         'seed': 2025,
         'embedding_dim': 64,
         'layers': 3,
         'reg_weight': 1e-4,
         'ssl_reg': 0.1,
         'ssl_temp': 0.5,
         'ssl_ratio': 0.2,
         'aug_type': 'ed'
    },
    'SimGCL': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 2048,
        'batch_user': 256,
        'seed': 2025,
        'embedding_dim': 64,
        'layers': 3,
        'reg_weight': 1e-4,
        'ssl_reg': 0.1,
        'ssl_temp': 0.2,
        'eps': 0.2
    },
    'LightGCL': {
        'epochs': 50,
        'lr': 1e-3,
        'batch_size': 4096,
        'batch_user': 256,
        'seed': 2025,
        'embedding_dim': 64,
        'layers': 2,
        'reg_weight': 1e-7,
        'ssl_reg': 0.2,
        'ssl_temp': 0.2,
        'ssl_ratio': 0.0,
        'svd_q': 5,
    },
    'AdaMCL': {
        'epochs': 300,
        'lr': 1e-3,
        'batch_size': 4096,
        'batch_user': 256,
        'seed': 2025,
        'embedding_dim': 64,
        'layers': 3,
        'reg_weight': 1e-4,
        'ssl_reg': 1e-6,
        'ssl_temp': 0.2,
        'layer_reg': 1e-6,
        'alpha': 1,
        'threshold': 0.8,
        'topk': 5,
        'gamma': 1,
    },
}

def get_config():
    parser = argparse.ArgumentParser(description="Recommendation models experiment")
    # 공통 인자
    parser.add_argument('--dataset', type=str, default='gowalla', help="Dataset name")
    parser.add_argument('--model', type=str, default='bpr', help="Model type (e.g., 'bpr', 'lightgcn')")
    # model별 인자
    parser.add_argument('--epochs', type=int, default=None, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=None, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size")
    parser.add_argument('--batch_user', type=int, default=None, help="Batch user size for evaluation")
    parser.add_argument('--seed', type=int, default=None, help="Random seed")
    parser.add_argument('--embedding_dim', type=int, default=None, help="Embedding dimension")
    parser.add_argument('--layers', type=int, default=None, help="Propagation layers (for LightGCN)")
    parser.add_argument('--reg_weight', type=float, default=None, help="Regularization weight (lambda_2)")
    parser.add_argument('--ssl_reg', type=float, default=None, help="Regularization of SSL loss (lambda_1)")
    parser.add_argument('--ssl_temp', type=float, default=None, help="Temperature for InfoNCE loss (tau)")
    parser.add_argument('--ssl_ratio', type=float, default=None, help="Node dropout ratio (rho)")
    parser.add_argument('--aug_type', type=float, default=None, help="Graph agmentation type")
    
    parser.add_argument('--eps', type=float, default=None, help="Epsilon of SimGCL")
    
    parser.add_argument('--svd_q', type=float, default=None, help="Rank of LightGCL SVD")
    
    # AdaMCL
    parser.add_argument('--layer_reg', type=float, default=None, help="Regularization weight of Layer contrastive loss")
    parser.add_argument('--alpha', type=float, default=None, help="Graph Contrastive loss summation weight for AdaMCL")
    parser.add_argument('--threshold', type=float, default=None, help="Graph G hat construction filter threshold for AdaMCL")
    parser.add_argument('--topk', type=float, default=None, help="Graph G hat construction filter topk for AdaMCL")
    parser.add_argument('--gamma', type=float, default=None, help="Gamma of beta for AdaMCL")
    
    
    
    args = parser.parse_args()
    model = args.model
    defaults = DEFAULTS.get(model, {})
    if args.epochs is None:
        args.epochs = defaults.get('epochs', 1000)
    if args.lr is None:
        args.lr = defaults.get('lr', 1e-3)
    if args.batch_size is None:
        args.batch_size = defaults.get('batch_size', 2048)
    if args.batch_user is None:
        args.batch_user = defaults.get('batch_user', 256)
    if args.seed is None:
        args.seed = defaults.get('seed', 2020)
    if args.embedding_dim is None:
        args.embedding_dim = defaults.get('embedding_dim', 64)
    if args.layers is None:
        args.layers = defaults.get('layers', 0)
    if args.reg_weight is None:
        args.reg_weight = defaults.get('reg_weight', 1e-4)
    if args.ssl_reg is None:
        args.ssl_reg = defaults.get('ssl_reg', 0.1)
    if args.ssl_temp is None:
        args.ssl_temp = defaults.get('ssl_temp', 0.5)
    if args.ssl_ratio is None:
        args.ssl_ratio = defaults.get('ssl_ratio', 0.2)
    if args.aug_type is None:
        args.aug_type = defaults.get('aug_type', 'ed')
    if args.eps is None:
        args.eps = defaults.get('eps', 0.2)
    if args.svd_q is None:
        args.svd_q = defaults.get('svd_q', 5)
    if args.layer_reg is None:
        args.layer_reg = defaults.get('layer_reg', 1e-6)
    if args.alpha is None:
        args.alpha = defaults.get('alpha', 1)
    if args.threshold is None:
        args.threshold = defaults.get('threshold', 0.8)
    if args.topk is None:
        args.topk = defaults.get('topk', 5)
    if args.gamma is None:
        args.gamma = defaults.get('gamma', 1)

    return vars(args)