from abc import ABC, abstractmethod


class Reader(ABC):

    def __init__(self, splits):
        """Initialize the reader.

        splits - dict
            A dictionary of {split_name: int} which maps the name of the split
            to the number of samples in that split
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
