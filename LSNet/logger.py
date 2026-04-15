from tqdm import tqdm
import sys

class MyTqdm(tqdm):
    def finish(self):
        self.close()

class TermLogger(object):
    def __init__(self, n_epochs, train_size, valid_size):
        self.n_epochs = n_epochs
        self.train_size = train_size
        self.valid_size = valid_size
        self.train_bar = None
        self.valid_bar = None

    def reset_train_bar(self):
        self.train_bar = MyTqdm(total=self.train_size, desc='Training',
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    def reset_valid_bar(self):
        self.valid_bar = MyTqdm(total=self.valid_size, desc='Validation',
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    @property
    def train_writer(self):
        class TrainWriter:
            @staticmethod
            def write(string):
                tqdm.write(string)
        return TrainWriter()

    @property
    def valid_writer(self):
        class ValidWriter:
            @staticmethod
            def write(string):
                tqdm.write(string)
        return ValidWriter()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0] * i
        self.avg = [0] * i
        self.sum = [0] * i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert (len(val) == self.meters)
        self.count += n
        for i, v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)