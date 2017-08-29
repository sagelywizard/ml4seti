import os

import ibmseti
import torch
import torch.multiprocessing as multiprocessing

from util import LABEL_TO_ID

def parse_dat(directory, cache, guid, target):
    if os.path.isfile('%s%s.pth' % (directory, guid)):
        tensor = torch.load('%s%s.pth' % (directory, guid))
        return tensor, LABEL_TO_ID[target]
    else:
        raw_file = open('%s%s.dat' % (directory, guid), 'rb')
        aca = ibmseti.compamp.SimCompamp(raw_file.read())
        spectrogram = aca.get_spectrogram()
        tensor = torch.from_numpy(spectrogram).float().view(1, 384, 512)
        if cache:
            torch.save(tensor, '%s%s.pth' % (directory, guid))
        return tensor, LABEL_TO_ID[target]

class Subset(object):
    def __init__(self, directory, dataset, start, end, pool_size=8, cache=False,
            minibatch_size=10):
        self.directory = directory
        self.cache = cache
        self.minibatch_size = minibatch_size
        self.size = len(dataset)
        self.start = int(self.size * start)
        self.end = int(self.size * end)
        self.pool = multiprocessing.Pool(pool_size)
        self.subset = dataset[self.start:self.end]
        self.index = 0

        self.iter = iter(self.subset)

    def progress(self):
        return self.index, self.end-self.start

    def reload(self):
        """Make the subset iterable again, from the beginning"""
        self.index = 0
        self.iter = iter(self.subset)

    def __iter__(self):
        return self

    def __next__(self):
        guids = []
        try:
            for _ in range(self.minibatch_size):
                guids.append(next(self.iter))
        except StopIteration:
            if guids == []:
                raise
        ret = self.pool.starmap(
            parse_dat,
            [(self.directory, self.cache, guid, target) for guid, target in guids])
        self.index += len(ret)
        return zip(*ret)

class Dataset(object):
    """This is an object which takes a directory containing the full dataset.
    When iterated over, this object returns (minibatch, minibatch target) pairs.
    """
    def __init__(self, directory, split=(0.6, 0.2, 0.2), minibatch_size=10,
            pool_size=8, cache=False, index_file):
        if not directory.endswith('/'):
            self.directory = '%s/' % directory
        else:
            self.directory = directory

        lines = open(index_file, 'r').readlines()[1:]
        dataset = [tuple(line.strip().split(',', 1)) for line in lines]

        train, valid, test = split
        self.train = Subset(
            self.directory,
            dataset,
            0.0,
            train,
            minibatch_size=minibatch_size,
            pool_size=pool_size,
            cache=cache)
        self.validate = Subset(
            self.directory,
            dataset,
            train,
            train+valid,
            minibatch_size=minibatch_size,
            pool_size=pool_size,
            cache=cache)
        self.test = Subset(
            self.directory,
            dataset,
            train+valid,
            train+valid+test,
            minibatch_size=minibatch_size,
            pool_size=pool_size,
            cache=cache)
