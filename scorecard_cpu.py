#!/usr/bin/env python3
"""A script for classifying a single sample using my model for the ml4seti competition.

e.g. python scorecard_cpu.py /path/to/data/files /path/to/model.pth output_scorecard.csv
"""
import sys
import glob
import os
import csv

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
    return Variable(tensor, volatile=True).cpu(), aca.header()

def get_densenet(model_path):
    dense = DenseNet(False).cpu()
    dense.eval()
    state = torch.load(model_path, map_location=lambda storage, loc: storage)
    dense.load_state_dict(state['model'])
    return dense.cpu()

def main(data_dir, model_path, output_scorecard):

    model = get_densenet(model_path)
    datafiles = glob.glob( os.path.join(data_dir, "*.dat") )

    with open(output_scorecard, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')

        for filename in datafiles:
            spec, header = get_spectrogram(filename)
            results = F.softmax(model(spec)).data.view(7).tolist()
            results.insert(0, header['uuid']) 
            print(results)
            csvwriter.writerow(results) 


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])