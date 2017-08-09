#!/usr/bin/env python3
"""
An entry into the ml4seti signal classifier competition.

This entry is simply a large densenet architecture convolutional neural
network. For more information, see "Densely Connected Convolutional Networks"
<https://arxiv.org/pdf/1608.06993.pdf>
"""
import argparse

import sklearn.metrics
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim.lr_scheduler

from dataset import Dataset
from model import DenseNet

class Experiment(object):
    def __init__(self, directory, epochs=1):
        self.dataset = Dataset(directory)
        self.epochs = epochs
        self.model = DenseNet()

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_fun = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            verbose=True,
            patience=3
        )
        for epoch in self.epochs:
            self.model.train()
            for minibatch, targets in self.dataset.train:
                out = self.model.forward(minibatch)
                loss = loss_fun(out, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            self.model.eval()
            for minibatch, targets in self.dataset.validate:
                out = self.model.forward(minibatch)
                validation_loss = loss_fun(out, targets)

    def test(self):
        confusion_matrix = np.zeros((7, 7)).astype(np.int)
        for minibatch, targets in self.dataset.test:
            out = self.model.forward(minibatch)
            _, predicted = torch.max(out.data, 1)
            predicted = predicted.cpu().numpy()
            targets = targets.cpu().numpy()
            confusion_matrix += sklearn.metrics.confusion_matrix(
                predicted,
                targets,
                labels=[0, 1, 2, 3, 4, 5, 6]
            ).astype(np.int)
        print(confusion_matrix)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory containing full dataset')
    parser.add_argument(
        '-e',
        '--epochs',
        type=int,
        default=1,
        help='Number of epochs to train')
    args = parser.parse_args()

    experiment = Experiment(args.directory, epochs=args.epochs)
    experiment.train()

if __name__ == '__main__':
    main()
