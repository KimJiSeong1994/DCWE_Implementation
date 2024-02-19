import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class DBEModel(nn.Module) :
    def __init__(self, V, T, m_t, dict, sampling_distribution, k = 50, lambda_ = 1e4, lambda_0 = 1, ns = 20) :
        super().__init__()
        self.V = V # -- vocab size
        self.T = T
        self.k = k # -- embedding dim.

        self.total_tokens = sum(m_t.values())
        self.lambda_ = lambda_ # -- Scalling factor on time drift prior
        self.lambda_0 = lambda_0 # -- Scalling factor on embedding prior

        self.sampling_distribution = Categorical(logits = sampling_distribution)
        self.negative_samples = ns
        self.dictionary = dict
        self.dictionary_reverse = {v : k for k, v in dict.items()}

        self.rho = nn.Embedding(V * T, k) # -- stacked dynamic embedding
        self.alpha = nn.Embedding(V, k) # -- Time independent context embedding
        with torch.no_grad() :
            nn.init.normal_(self.rho.weight, 0, .01)
            nn.init.normal_(self.alpha.weight, 0, .01)

        self.log_sigmoid = nn.LogSigmoid()
        self.sigmoid = nn.Sigmoid()

    def L_pos(self, eta) :
        return self.log_sigmoid(eta).sum()

    def L_neg(self, batch_size, times, context_summed) :
        neg_samples = self.sampling_distribution.sample(torch.Size([batch_size, self.negative_samples]))
        neg_samples = neg_samples + (times * self.V).reshape((-1, 1))
        neg_samples = neg_samples.T.flatten()

        context_flat = context_summed.repeat((self.negative_samples, 1))
        eta_neg = (self.rho(neg_samples) * context_flat).sum(axis = 1)
        return (torch.log(1 - self.sigmoid(eta_neg) + 1e-7)).sum()

    def forward(self, targets, times, contexts, validate = False, dynamic = True) :
        batch_size = targets.shape[0]
        targets_adjusted = times * self.V + targets

        context_mask = contexts == -1
        contexts[context_mask] = 0
        contexts = self.alpha(contexts)
        contexts[context_mask] = 0
        contexts_summed = contexts.sum(axis = 1)
        eta = (self.rho(targets_adjusted) * contexts_summed).sum(axis = 1)

        loss, L_pos, L_neg, L_prior = None, None, None, None
        L_pos = self.L_pos(eta) # nomalized (scaling vector)
        if not validate :
            L_neg = self.L_neg(batch_size, times, contexts_summed)
            loss = (self.total_tokens / batch_size) * (L_pos + L_neg)
            L_prior = - self.lambda_0 / 2 * (self.alpha.weight ** 2).sum()
            L_prior += - self.lambda_0 / 2 * (self.rho.weight[0] ** 2).sum()
            if dynamic :
                rho_trans = self.rho.weight.reshape((self.T, self.V, self.k))
                L_prior += (-self.lambda_ / 2 * ((rho_trans[1:] - rho_trans[:-1]) ** 2).sum())

            loss += L_prior
            loss = -loss

        return loss, L_pos, L_neg, L_prior

    def get_embeddings(self) :
        return (self.rho.cpu().weight.data.reshape((self.T, len(self.dictionary), self.k)).numpy())
        
