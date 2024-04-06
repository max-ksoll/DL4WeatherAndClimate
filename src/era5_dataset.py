import enum

import numpy as np
import torch
import zarr


class TimeMode(enum.Enum):
    ALL = 0
    AFTER = 1
    BEFORE = 2
    BETWEEN = 3


class ERA5Dataset(torch.utils.data.IterableDataset):

    def __init__(self, path_file, batch_size,
                 time_mode: TimeMode,
                 max_autoregression_steps=1,
                 start_time="2011-01-01T00:00:00",
                 end_time="2011-12-31T18:00:00"):
        super(ERA5Dataset, self).__init__()

        self.batch_size = batch_size

        store = zarr.DirectoryStore(path_file)
        self.sources = zarr.group(store=store)

        self.mins = torch.Tensor(
            [193.48901, -3.3835982e-05, -65.45247, -96.98215, -6838.8906]
        )
        self.maxs = torch.Tensor(
            [324.80637, 0.029175894, 113.785934, 89.834595, 109541.625]
        )
        self.max_minus_min = self.maxs - self.mins
        self.mins = self.mins[:, None, None, None]
        self.max_minus_min = self.max_minus_min[:, None, None, None]
        self.max_autoregression_steps = max_autoregression_steps + 2

        times = np.array(self.sources["time"])
        if time_mode == TimeMode.AFTER:
            times = times >= np.datetime64(start_time)
        elif time_mode == TimeMode.BEFORE:
            times = times <= np.datetime64(end_time)
        elif time_mode == TimeMode.BETWEEN:
            times_gt = times >= np.datetime64(start_time)
            times_ls = times <= np.datetime64(end_time)
            times = times_gt & times_ls
        else:
            times = np.ones_like(times)

        self.idxs = np.arange(self.sources["time"].shape[0])[times]
        if len(self.idxs) > 0:
            keep_idxs = self.idxs <= np.max(self.idxs)-self.max_autoregression_steps
            self.idxs = self.idxs[keep_idxs]
        self.len = self.idxs.shape[0]

        self.rng = np.random.default_rng()

    def shuffle(self):
        self.idxs = self.rng.permutation(self.idxs)

    def __len__(self):
        return self.len

    def __iter__(self):

        self.shuffle()
        iter_start, iter_end = self.worker_workset()

        for bidx in range(iter_start, iter_end, self.batch_size):
            idx_t = self.idxs[bidx: bidx + self.batch_size]

            idxes = [idx_t + i for i in range(self.max_autoregression_steps)]
            sources = [
                self.get_at_idx(idx) for idx in idxes
            ]
            targets = sources[2:]
            sources = sources[:-1]

            source = torch.stack(sources, dim=1)
            target = torch.stack(targets, dim=1)

            # Normalization
            source = (source - self.mins) / self.max_minus_min
            target = (target - self.mins) / self.max_minus_min

            source = source.flatten(start_dim=2, end_dim=3)
            target = target.flatten(start_dim=2, end_dim=3)

            yield source, target

    def get_at_idx(self, idx_t):
        return torch.stack(
            [
                torch.tensor(self.sources["t"][idx_t]),
                torch.tensor(self.sources["q"][idx_t]),
                torch.tensor(self.sources["u"][idx_t]),
                torch.tensor(self.sources["v"][idx_t]),
                torch.tensor(self.sources["z"][idx_t]),
            ],
            1,
        )

    def __len__(self):
        return self.len // self.batch_size

    #####################
    def worker_workset(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            iter_start = 0
            iter_end = len(self) - self.max_autoregression_steps

        else:
            # split workload
            temp = len(self)
            per_worker = int(np.floor(temp / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = int(worker_id * per_worker)
            iter_end = int(iter_start + per_worker)
            if worker_info.id + 1 == worker_info.num_workers:
                iter_end = int(temp) - self.max_autoregression_steps

        return iter_start, iter_end
