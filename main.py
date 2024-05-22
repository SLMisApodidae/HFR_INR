import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import scipy.io
from dataio import *
from models import *
from util import *
from train import *
import argparse
from hash_encoding import HashEncoder
parser = argparse.ArgumentParser(description='PyTorch Implementation of SSR-TVD')
parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                    help='learning rate')
parser.add_argument('--batch_size', type=int, default=20000, metavar='N',
                    help='input batch size for training')
parser.add_argument('--dataset', type=str, default = 'flowDM',
                    help='dataset')
parser.add_argument('--mode', type=str, default='train' ,
                    help='training or inference')
parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                    help='number of epochs to train (default: 400)')

args = parser.parse_args()

def main(args):
	seed = 0
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	
	if args.mode == 'train':
		dtrain = RT3_16X(460, 481,train=True)
		dtest = RT3_16X(500, 501)
		dataloader_train = torch.utils.data.DataLoader(dtrain, batch_size=args.batch_size, shuffle=True, num_workers=1)
		dataloader_test = torch.utils.data.DataLoader(dtest, batch_size=args.batch_size, shuffle=False, num_workers=1)
		bounding_box=torch.tensor(([0,0,0.36],[0.25,1,0.96])).to('cuda')
		hash_encoder = HashEncoder(bounding_box)
		hash_encoder.to('cuda')
		model = FlashMHATrans(hash_encoder,d_out=1, d_hidden=256, d_model=32, N=1, heads=2).to('cuda')
		# model.apply(init_weights)
		trainNet(model,dataloader_train, args,scheduler = nifll_scheduler, validation_data=dataloader_test, save_model_freq=1000)

	# elif args.mode == 'inf':
	
if __name__== "__main__":
    main(args)





