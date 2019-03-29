from abc import ABC, abstractmethod
import collections

import numpy as np
import torch
from torch.utils import data


class Reader(ABC):

    def __init__(self, splits):
        """Initialize the reader.

        splits - dict
            A dictionary of {split_name: int} which maps the name of the split to the number of samples in that split
        """
        self.splits = splits

    def get_size(self, split=None):
        if split is not None:
            return self.splits[split]
        else:
            return sum(self.splits.values())

    @abstractmethod
    def iter_batches(self, split_name, batch_size, shuffle=True, partial_batching=False, threads=1, epochs=1,
                     max_batches=-1):
        pass


class DatasetReader(Reader):
    """Creates a Reader wrapper around PyTorch datasets."""

    def __init__(self, dataset_splits):
        """dataset_splits is a dictionary of {split_name: Dataset}"""
        self.dataset_splits = dataset_splits
        super().__init__(collections.OrderedDict([(k, len(v)) for k, v in dataset_splits.items()]))
        np.random.seed(0)

    def process_batch(self, batch):
        """Process batch before yielding from iter_batches."""
        return batch

    def iter_batches(self, split_name, batch_size, shuffle=True, partial_batching=False, threads=1, epochs=1,
                     max_batches=-1):
        rng_state = torch.get_rng_state()
        loader = data.DataLoader(self.dataset_splits[split_name], batch_size=batch_size, shuffle=shuffle,
                                 num_workers=threads, drop_last=not partial_batching)
        torch.set_rng_state(rng_state)  # don't let DataLoader reset rng
        for _ in range(epochs):
            for i, batch in enumerate(loader):
                yield self.process_batch(batch)
                if i + 1 == max_batches:
                    break
