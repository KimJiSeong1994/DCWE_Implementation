
class Train :
    def __init__(self, dataset, dict, validation = None, m = 300, num_epochs = 10, lr = 2e-3, validate_after = 100, **kwargs) :
        import torch
        import numpy as np
        from tqdm import tqdm

        from src.DataLoader import GetData
        from src.Model import DBEModel

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        validation_mask = np.repeat(False, dataset.shape[0])
        if validation is not None :
            assert 0 < validation < 1
            validation_mask = np.random.random(dataset.shape[0]) < validation

        data = GetData(dataset[~validation_mask], dict, device)
        data = GetData(dataset[validation_mask], dict, device)

        model = DBEModel(len(data.dictionary), data.T, data.m_t, dict, data.unigram_logits, **kwargs,)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
