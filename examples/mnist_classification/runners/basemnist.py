import importlib

from readers.mnist import MNISTReader
from pylego import misc, runner


class MNISTBaseRunner(runner.Runner):

    def __init__(self, flags, model_class, *args, **kwargs):
        self.flags = flags
        reader = MNISTReader(flags.data_path)
        summary_dir = flags.log_dir + '/summary'
        super().__init__(reader, flags.batch_size, flags.epochs, summary_dir, threads=flags.threads,
                         print_every=flags.print_every, visualize_every=flags.visualize_every,
                         max_batches=flags.max_batches, *args, **kwargs)
        model_class = misc.get_subclass(importlib.import_module('models.' + self.flags.model), model_class)
        self.model = model_class(self.flags, optimizer=flags.optimizer, learning_rate=flags.learning_rate,
                                 cuda=flags.cuda, load_file=flags.load_file, save_every=flags.save_every,
                                 save_file=flags.save_file, debug=flags.debug)
