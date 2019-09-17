import collections

import torch

from models.baseclf import BaseClassification
from .basemnist import MNISTBaseRunner


class ClassificationRunner(MNISTBaseRunner):

    def __init__(self, flags, *args, **kwargs):
        super().__init__(flags, BaseClassification)

    def run_batch(self, batch, train=False):
        x, y = self.model.prepare_batch(batch)
        loss = self.model.run_loss(x, labels=y)
        if train:
            self.model.train(loss, clip_grad_norm=self.flags.grad_norm)

        return collections.OrderedDict([('loss', loss.item())])

    def post_epoch_visualize(self, epoch, split):
        if split == 'train' and not self.flags.visualize_only:
            return
        print('* Evaluating', split)
        total = 0
        correct = 0
        for batch in self.reader.iter_batches(split, self.flags.batch_size, shuffle=False, partial_batching=True,
                                              threads=self.threads):
            x, y = self.model.prepare_batch(batch)
            y_logits = self.model.run_batch([x])
            y_pred = torch.argmax(y_logits, dim=1)
            total += y_pred.size(0)
            correct += (y_pred == y).sum().item()

        acc = 100.0 * correct / total
        print('* %s accuracy: %.2f%%' % (split, acc))
