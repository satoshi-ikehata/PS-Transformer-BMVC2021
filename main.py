from __future__ import print_function, division
import torch
from modules.io.custom_dataloader import *
from modules.builder import builder
from modules.builder import logger
import sys
sys.path.append('..') 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--session_name', default='eval')
parser.add_argument('--train_num_samples', type=int, default=50000)
# parser.add_argument('--cycles', default='None')
parser.add_argument('--diligent', default='None')
parser.add_argument('--pretrained', default='./pretrained')
parser.add_argument('--checkpoint', default='./checkpoint')
parser.add_argument('--n_testimg', type=int, default=10)


def main():
    args = parser.parse_args()
    outdir = '.'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # trainData = custom_dataloader('Cycles', args.cycles, num_samples=args.train_num_samples)
    testData = custom_dataloader('DiLiGenT', args.diligent, min_nimg=args.n_testimg, max_nimg=args.n_testimg)
    trainObj = builder.builder(device)
    if args.pretrained is not None:
        trainObj.net.load_models(args.pretrained)
    tensorboard_writer = logger.logger(outdir, f'Tensorboard/{args.session_name}')

    epochs = 1
    for epoch in range(epochs):
        print(f'Run {epoch}-th epoch')
        print(f'Number of Test Sets {testData.data.get_num_set()}, Number of Objs {testData.data.get_num_object()}')
        losses = 0
        errors = 0
        
        """TEST"""
        outputs = []
        errors = 0
        for k in range(testData.data.get_num_object()):
            testData.reset()
            testData.load(objid=k)
            output, error = trainObj.run('Test', data=testData)
            outputs.append(output)
            errors += error
            print(f'Error {error:.02f}')

        outputs = torch.cat(outputs, dim=3)
        tensorboard_writer.add('[Test] Normal Maps', outputs, epoch, 'Image')
        tensorboard_writer.add('[Test] Error', errors, epoch, 'Scalar')
        print(f"[TEST] Mean Error {errors/testData.data.get_num_object():.02f} deg.")


main()
