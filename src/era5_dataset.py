import numpy as np
import torch
import zarr
import pandas as pd
import code


class ERA5Dataset(torch.utils.data.IterableDataset):

    def __init__(self, path_file, batch_size):
        super(ERA5Dataset, self).__init__()

        self.batch_size = batch_size

        store = zarr.DirectoryStore(path_file)
        self.sources = zarr.group(store=store)

        self.mins = torch.Tensor([193.48901, -3.3835982e-05, -65.45247, -96.98215, -6838.8906])
        self.maxs = torch.Tensor([324.80637, 0.029175894, 113.785934, 89.834595, 109541.625])
        self.max_minus_min = (self.maxs - self.mins)
        self.mins = self.mins[:, None, None, None]
        self.max_minus_min = self.max_minus_min[:, None, None, None]

        self.rng = np.random.default_rng()
        self.shuffle()

    def shuffle(self):

        len = self.sources['time'].shape[0]
        self.idxs = self.rng.permutation(np.arange(len))

        self.len = self.idxs.shape[0]

    def __len__(self):
        return self.len

    def __iter__(self):

        self.shuffle()
        iter_start, iter_end = self.worker_workset()

        for bidx in range(iter_start, iter_end, self.batch_size):
            idx_t = self.idxs[bidx: bidx + self.batch_size]

            source_t_m1 = self.get_at_idx(idx_t - 1)
            source_t = self.get_at_idx(idx_t)

            source = torch.stack([source_t_m1, source_t], dim=1)
            target = self.get_at_idx(idx_t + 1)

            # Normalization
            source = (source - self.mins) / self.max_minus_min
            target = (target - self.mins) / self.max_minus_min

            source = source.flatten(start_dim=2, end_dim=3)
            target = target.flatten(start_dim=1, end_dim=2)

            yield source, target

    def get_at_idx(self, idx_t):
        return torch.stack([
            torch.tensor(self.sources['t'][idx_t]),
            torch.tensor(self.sources['q'][idx_t]),
            torch.tensor(self.sources['u'][idx_t]),
            torch.tensor(self.sources['v'][idx_t]),
            torch.tensor(self.sources['z'][idx_t])], 1)

    def __len__(self):
        return self.len // self.batch_size

    #####################
    def worker_workset(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            iter_start = 1
            iter_end = len(self)

        else:
            # split workload
            temp = len(self)
            per_worker = int(np.floor(temp / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = int(worker_id * per_worker)
            if iter_start == 0:
                iter_start = 1
            iter_end = int(iter_start + per_worker)
            if worker_info.id + 1 == worker_info.num_workers:
                iter_end = int(temp)

        return iter_start, iter_end
