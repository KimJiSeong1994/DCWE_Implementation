import torch
from torch.utils.data import Dataset

class MLMDataset(Dataset) :
    def __init__(self, name, split, social_dim, data_dir) :
        from transformers import BertTokenizer
        self.tok = BertTokenizer.from_pretrained('bert-base-uncased')

