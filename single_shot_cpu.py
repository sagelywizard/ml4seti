#!/usr/bin/env python3
"""A script for classifying a single sample using my model for the ml4seti competition.

e.g. python single_shot_cpu.py /path/to/sample.dat /path/to/model.pth
"""
import sys

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import ibmseti  #need version 2.0.0 or greater (lastest version: pip install ibmseti==2.0.0.dev5)
import time

from model import DenseNet

def get_spectrogram(filename):
    raw_file = open(filename, 'rb')
    aca = ibmseti.compamp.SimCompamp(raw_file.read())
    tensor = torch.from_numpy(aca.get_spectrogram()).float().view(1, 1, 384, 512).cpu()
    return Variable(tensor, volatile=True).cpu()

def get_densenet(model_path):
    dense = DenseNet(False).cpu()
    dense.eval()
    state = torch.load(model_path, map_location=lambda storage, loc: storage)
    dense.load_state_dict(state['model'])
    return dense.cpu()

def main(filename, model_path):

    cudnn.benchmark=False

    st = time.time()
    spec = get_spectrogram(filename)
    print('time to build spectrogram: {}'.format(time.time() - st))
    st = time.time()
    model = get_densenet(model_path)
    print('time to load model: {}'.format(time.time() - st))
    st = time.time()
    results = F.softmax(model(spec)).data.view(7)
    print('time to compute class probs: {}'.format(time.time() - st))
    print(results.tolist())

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])