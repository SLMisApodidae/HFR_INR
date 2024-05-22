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
import datetime

def saveFile2Dxyvali(outputFilePath,data,x,y,dim):
	# XYZ=getBXYZ_3D(dim)
	# _,x,y,z,_=coords.shape
	f=open(outputFilePath,"w")
	
	f.write(" variables=\"x\",\"y\",\"u\" \n" )
	f.write("zone i= "+ '{:4d}'.format(dim[1])+" ,j= "+'{:4d}\n'.format(dim[0]))
	index=0
	for j in range(dim[1]):
		for i in range(dim[0]):
			f.write("%.7f %.7f %.7f\n"%(float(y[index]),float(x[index]),float(data[index])))
			# f.write("%.7f %.7f %.7f %.7f\n"%(float(i),float(j),float(k),float(data[index])))
			index+=1
		
	f.close()

def get_mgrid(sidelen, dim=2, s=1,t=0):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)
    if dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]:(t+1),0:1:sidelen[2]*1j, 0:0.25:sidelen[1]*1j], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
    elif dim == 2:
        pixel_coords = np.stack(np.mgrid[0:1:sidelen[1]*1j, 0:0.25:sidelen[0]*1j], axis=-1)[None, ...].astype(np.float32)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords = np.reshape(pixel_coords,(-1,dim))
    return pixel_coords

def validate(model,criterion, dataloader,saveResult):
	epoch_loss = 0
	v = []
	x = []
	y = []


	for idx, batch in enumerate(dataloader):
		batch_t, batch_x, batch_y, batch_data= batch
		batch_x = batch_x.cuda()
		batch_y = batch_y.cuda()
		batch_t = batch_t.cuda()
		batch_data = batch_data.cuda()
		
		# Compute the loss the for the discriminator with real images
		# self.model.zero_grad()
		vhat = model(batch_x,batch_y,batch_t)
		vhat = vhat[:,0,0]
		vT = batch_data[:,0,0]
		xhat = batch_x[:,0,0]
		yhat = batch_y[:,0,0]

		# print(vhat.shape, " ",vT.shape)
		loss = criterion(vhat.view(-1), vT.view(-1))
		v += list(vhat.view(-1).detach().cpu().numpy())
		x += list(xhat.view(-1).detach().cpu().numpy())	
		y += list(yhat.view(-1).detach().cpu().numpy())

		# Update running losses
		epoch_loss += loss.mean().item()
		# if self.verbose == 1:
			# print_progress_bar(idx, len(dataloader))
		
	v = np.asarray(v,dtype='<f')
	x = np.asarray(x,dtype='<f')
	y = np.asarray(y,dtype='<f')
	v=(v+1)*2.4/2-1.2
	dataid=500
	data_pathLow = '/home/nvme0/dx/dl/weno_code/weno3/RT480/'#'../../../../nvme0/dx/dl/weno_code/weno3/DM480/'
	d = getBData_2D(data_pathLow+'RESU'+'{:04d}'.format(dataid)+'.DAT',[120,480])[0]
	d = resize(d,(480,1920),order=3)
	d = d.flatten('F')
	result=d+v
	outputFilePath1='./logs/{:04d}'.format(dataid)+'.plt'
	outputFilePath='./logs/Delta{:04d}'.format(dataid)+'.plt'
	# print(coords.shape)
	coords = get_mgrid([480,1920],dim=2,t=0)

	saveFile2Dxyvali(outputFilePath1,result,x,y,[480,1920])
	saveFile2Dxyvali(outputFilePath,v,x,y,[480,1920])

	return epoch_loss

def train_epoch(model, dataloader,opt,criterion,scheduler,args,n):

	epoch_loss  = 0
	print("begin _train_epoch")
	# with autograd.detect_anomaly():
	model=model.cuda()
	for idx, batch in enumerate(dataloader):
		batch_t, batch_x, batch_y, batch_data= batch
		batch_x = batch_x.cuda()
		batch_y = batch_y.cuda()
		batch_t = batch_t.cuda()
		batch_data = batch_data.cuda()
		
		
		yhat = model(batch_x,batch_y,batch_t)
		# print(yhat.shape," ",batch_data.shape)
		loss = criterion(yhat.view(-1), batch_data.view(-1))
		opt.zero_grad()
		loss.backward()
		opt.step()
		# print(idx," ",lossap)
		epoch_loss +=loss.data.mean().item()
		
		
		# opt.param_groups[0]['lr'] = scheduler(n, args.lr)
	
		# # Update running losses
		# epoch_loss += loss.item()
	print("end _train_epoch")
	return epoch_loss


def trainNet(model,dataloader_train, args,scheduler, validation_data, save_model_freq):
	optim = torch.optim.AdamW(model.parameters(),lr=args.lr)
	#LBFGS(model.parameters(),lr=args.lr,line_search_fn='strong_wolfe')
	n_params = get_n_params(model)
	print(model)
	print(get_n_params(model))
	criterion = nn.MSELoss(reduction='mean')
	start_time = datetime.datetime.now()
	
	loss = open('./loss.txt','w')
	loss_track = []
	scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim,mode='min',factor=0.1,patience=20,verbose=True,threshold=0.0001,threshold_mode='rel',cooldown=0,eps=1e-9)
	for n in range(1,args.epochs+1):

		t_loss = train_epoch(model,dataloader_train,optim,criterion,scheduler,args,n)
		
		print(f'Train one epoch in {str(datetime.datetime.now() - start_time).split(".")[0]}')
		v_loss = 0
		if n%20==0:
			saveResult=True
			with torch.no_grad():
				v_loss = validate(model,criterion,validation_data,saveResult)
		print(n," train_loss: ",t_loss," val_loss: ",v_loss)
		loss.write("Epochs "+str(n)+": train_loss = "+str(t_loss)+": val_loss = "+str(v_loss))
		loss.write('\n')
		# epoch_result = {'train_loss': t_loss, 'val_loss': v_loss, save_criterion:t_loss}
		if n % save_model_freq == 0 :
			torch.save(model.state_dict(),'logs/'+str(n)+'.pth')
		scheduler.step(t_loss)

	loss.close()
	print(f'Training completed in {str(datetime.datetime.now() - start_time).split(".")[0]}')
	
