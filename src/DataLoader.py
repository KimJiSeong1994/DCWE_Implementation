class GetData :
    def __init__(self, df, dict, device) :
        import torch
        import numpy as np
        import pandas as pd
        from src.args import args
        from collections import Counter

        self.cs = args.cs
        self.dictionary = dict
        self.N = df.shape[0]
        self.ctx = None

        bow_filtered = df[args.bow_col].apply(lambda x : list(filter(lambda x : x is not None, [dict.get(w, None) for w in x])))
        tfs = Counter(word for row in bow_filtered for word in row)
        scaled_tfs = np.array([cnt for _, cnt in sorted(tfs.items())]) ** .75
        total = scaled_tfs.sum()

        self.unigram_logits = torch.tensor([np.log(cnt / (total - cnt)) for cnt in scaled_tfs]).to(device)
        df_idx = pd.DataFrame({'time' : df[args.time_col], 'bow' : bow_filtered})
        df_idx = df_idx[bow_filtered.apply(len) > 1]

        m_t = {}
        for t, group in df_idx.groupby(args.time_col) : m_t[t] = group[args.bow_col].apply(len).sum()
        self.m_t = m_t
        self.T = len(m_t)
        self.df_idx = df_idx

    def __len__(self) :
        return self.N

    def _context_mask(self, N) :
        import numpy as np
        if self.ctx is None or self.ctx.shape[0] < N : self.ctx = ( np.tile(np.arange(N), (self.cs * 2, 1)).T + np.delete(np.arange(2 * self.cs + 1) - self.cs))
        ctx = self.ctx[:N]
        oob = (ctx > N) | (ctx < 0)
        return ctx, oob

