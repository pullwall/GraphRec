import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix, diags

class RecDataset:
    def __init__(self, data_name):
        self.train, self.test = self.load_data(data_name)
        self.num_users, self.num_items = self.train.shape
        sparsity = 100 * (1 - (self.train.sum() + self.test.sum()) / (self.num_users * self.num_items))
        print(f"{data_name} data: {self.num_users} users, {self.num_items} items, sparsity: {sparsity:.4f}%")
        self.test_labels = self.create_test_labels()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")
        self.adj_matrix = self.create_adj_matrix()
        self.norm_adj_matrix = self.normalize_adj_matrix(self.adj_matrix)
        self.norm_adj_matrix = self.scipy_sparse_mat_to_torch_sparse_tensor(self.norm_adj_matrix)

    def load_data(self, data_name):
        path = './data/' + data_name + '/'
        def read_txt_to_coo(filename):
            user_ids = []
            item_ids = []
            with open(filename, 'r') as f:
                for line in f:
                    tokens = list(map(int, line.strip().split()))
                    if not tokens:
                        continue
                    user = tokens[0]
                    items = tokens[1:]
                    for item in items:
                        user_ids.append(user)
                        item_ids.append(item)
            num_users = max(user_ids) + 1
            num_items = max(item_ids) + 1
            data = [1] * len(user_ids)
            return coo_matrix((data, (user_ids, item_ids)), shape=(num_users, num_items))

        train = read_txt_to_coo(path + 'train.txt')
        test = read_txt_to_coo(path + 'test.txt')
        return train, test

    def create_test_labels(self):
        test_labels = [[] for _ in range(self.test.shape[0])]
        for i in range(len(self.test.data)):
            row = self.test.row[i]
            col = self.test.col[i]
            test_labels[row].append(col)
        print('Test data processed.')
        return test_labels

    def create_adj_matrix(self):
        R = self.train  # 이미 coo_matrix 형식
        num_users, num_items = self.num_users, self.num_items
        # 사용자-아이템, 아이템-사용자 edge 구성
        rows_ui = R.row
        cols_ui = R.col + num_users
        data_ui = R.data
        rows_iu = R.col + num_users
        cols_iu = R.row
        data_iu = R.data
        rows = np.concatenate([rows_ui, rows_iu])
        cols = np.concatenate([cols_ui, cols_iu])
        data = np.concatenate([data_ui, data_iu])
        adj_shape = (num_users + num_items, num_users + num_items)
        adj = coo_matrix((data, (rows, cols)), shape=adj_shape)
        return adj

    def normalize_adj_matrix(self, adj_matrix):
        row_sum = np.array(adj_matrix.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(row_sum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = diags(d_inv_sqrt)
        norm_adj = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
        return norm_adj.tocoo()

    def scipy_sparse_mat_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = sparse_mx.shape
        sparse_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
        return sparse_tensor.to(self.device)

class BatchDataLoader(Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]