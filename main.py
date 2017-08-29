#!/usr/bin/env python3
"""
An entry into the ml4seti signal classifier competition.

This entry is simply a large densenet architecture convolutional neural
network. For more information, see "Densely Connected Convolutional Networks"
<https://arxiv.org/pdf/1608.06993.pdf>
"""
import argparse
import time

import sklearn.metrics
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim.lr_scheduler
import tabulate

from dataset import Dataset
from model import DenseNet
from util import tprint, stats

class Experiment(object):
    def __init__(self, directory, epochs=1, cuda=False, save=False,
            log_interval=30, load=None, split=(0.6, 0.2, 0.2), cache=False,
            minibatch_size=10, pretrained=False, index_file=''):
        self.dataset = Dataset(directory, split=split, cache=cache,
            minibatch_size=minibatch_size, index_file=index_file)
        self.epochs = epochs
        self.cuda = cuda
        self.save = save
        self.log_interval = log_interval
        self.model = DenseNet(pretrained)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        if load is not None:
            state = torch.load(load)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optim'])
        if cuda:
            self.model = self.model.cuda()


    def train(self):
        print('Training %s epochs.' % self.epochs)
        loss_fun = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            'min',
            verbose=True,
            patience=3
        )
        last_print = time.time()
        for epoch in range(self.epochs):
            tprint('Starting epoch: %s' % epoch)
            self.model.train()
            self.optimizer.zero_grad()
            for minibatch, targets in self.dataset.train:
                minibatch = Variable(torch.stack(minibatch))
                targets = Variable(torch.LongTensor(targets))
                if self.cuda:
                    minibatch = minibatch.cuda()
                    targets = targets.cuda()
                out = self.model.forward(minibatch)
                loss = loss_fun(out, targets)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if time.time() - last_print > self.log_interval:
                    last_print = time.time()
                    numer, denom = self.dataset.train.progress()
                    tprint('Training: %s, %s/%s' % (epoch, numer, denom))
            tprint('Training complete. Beginning validation.')
            self.dataset.train.reload()
            self.model.eval()
            last_print = time.time()
            for minibatch, targets in self.dataset.validate:
                minibatch = Variable(torch.stack(minibatch), volatile=True)
                targets = Variable(torch.LongTensor(targets), volatile=True)
                if self.cuda:
                    minibatch = minibatch.cuda()
                    targets = targets.cuda()
                out = self.model.forward(minibatch)
                validation_loss = loss_fun(out, targets)
                if time.time() - last_print > self.log_interval:
                    last_print = time.time()
                    numer, denom = self.dataset.validate.progress()
                    tprint('Validating: %s, %s/%s' % (epoch, numer, denom))
            self.dataset.validate.reload()
            scheduler.step(validation_loss.data[0])
        if self.save:
            torch.save({
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict(),
            }, 'signet.%s.pth' % int(time.time()))

    def test(self):
        tprint('Beginning testing.')
        confusion_matrix = np.zeros((7, 7)).astype(np.int)
        last_print = time.time()
        for minibatch, targets in self.dataset.test:
            minibatch = Variable(torch.stack(minibatch), volatile=True)
            targets = Variable(torch.LongTensor(targets), volatile=True)
            if self.cuda:
                minibatch = minibatch.cuda()
                targets = targets.cuda()
            out = self.model.forward(minibatch)
            _, predicted = torch.max(out.data, 1)
            predicted = predicted.cpu().numpy()
            targets = targets.data.cpu().numpy()
            confusion_matrix += sklearn.metrics.confusion_matrix(
                predicted,
                targets,
                labels=[0, 1, 2, 3, 4, 5, 6]
            ).astype(np.int)
            if time.time() - last_print > self.log_interval:
                last_print = time.time()
                numer, denom = self.dataset.test.progress()
                tprint('Testing: %s/%s' % (numer, denom))
        tprint('Testing complete.')
        print(confusion_matrix)
        print(tabulate.tabulate(stats(confusion_matrix)))

def valid_split(arg):
    split = arg.split(',')
    if len(split) != 3:
        raise argparse.ArgumentTypeError("invalid split argument")
    try:
        train = float(split[0])
        valid = float(split[1])
        test = float(split[2])
    except ValueError:
        raise argparse.ArgumentTypeError("split args must be numbers")
    denom = train + valid + test
    return train/denom, valid/denom, test/denom

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory containing full dataset')
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=1,
        help='Number of epochs to train')
    parser.add_argument('-c', '--cuda', action='store_true', default=False)
    parser.add_argument(
        '--train',
        action='store_true',
        default=False,
        help='flag to signal script should train the model')
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help='flag to signal script should test the model')
    parser.add_argument(
        '-s',
        '--save',
        action='store_true',
        default=False, help='Will cause model to be saved at end of training')
    parser.add_argument(
        '-l',
        '--log-interval',
        type=int,
        default=30,
        help='# of seconds between log line prints')
    parser.add_argument(
        '-m',
        '--model',
        default=None,
        help='path to a pretrained model')
    parser.add_argument(
        '-p',
        '--split',
        default=(0.6, 0.2, 0.2),
        type=valid_split,
        help='train/validation/test set split')
    parser.add_argument(
        '--cache',
        default=False,
        action='store_true',
        help='flag to cache processed spectrograms')
    parser.add_argument(
        '-b',
        '--minibatch-size',
        type=int,
        default=10,
        help='size of each minibatch')
    parser.add_argument(
        '--pretrained',
        default=False,
        action='store_true',
        help='use DenseNet pretrained on ImageNet')
    parser.add_argument(
        '-i',
        '--index-file',
        type=str,
        default='public_list_primary_v3_full_21june_2017.csv',
        help='The index file for the data set, containing the UUID, class pairs.')

    args = parser.parse_args()
    if args.train or args.test:
        experiment = Experiment(
            args.directory,
            epochs=args.epochs,
            cuda=args.cuda,
            save=args.save,
            log_interval=args.log_interval,
            load=args.model,
            split=args.split,
            cache=args.cache,
            minibatch_size=args.minibatch_size,
            pretrained=args.pretrained,
            index_file=args.index_file)
        if args.train:
            experiment.train()
        if args.test:
            experiment.test()

if __name__ == '__main__':
    main()
