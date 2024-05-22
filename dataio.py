import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import data,img_as_float
import torch.nn.functional as F
import struct
import math
def get_mgrid(sidelen, dim=2, s=1,t=0):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]:(t+1),0:1:sidelen[2]*1j, 0:0.25:sidelen[1]*1j], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        # pixel_coords[..., 0]-=0.5
        # pixel_coords[..., 0]*=2.
        # pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[2] - 1)
        # pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[1] - 1)
    elif dim == 2:
        pixel_coords = np.stack(np.mgrid[0:1:sidelen[1]*1j, 0:0.25:sidelen[0]*1j], axis=-1)[None, ...].astype(np.float32)
        # pixel_coords[..., 0] = pixel_coords[..., 0] / (sidelen[1] - 1)
        # pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[0] - 1)
    elif dim == 4:
        pixel_coords = np.stack(np.mgrid[:sidelen[0]:(t+1), :sidelen[3]:s, :sidelen[2]:s, :sidelen[1]:s], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[3] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
        pixel_coords[..., 3] = pixel_coords[..., 3] / (sidelen[1] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    # print(pixel_coords.shape)
    pixel_coords = np.reshape(pixel_coords,(-1,dim))
    print(pixel_coords)
    return pixel_coords
def saveFile2D(outputFilePath,data,coords,dim):
	# XYZ=getBXYZ_3D(dim)
	print(coords.shape)
	# _,x,y,z,_=coords.shape
	f=open(outputFilePath,"w")
	
	f.write(" variables=\"x\",\"y\",\"u\" \n" )
	f.write("zone i= "+ '{:4d}'.format(dim[0])+" ,j= "+'{:4d}\n'.format(dim[1]))
	index=0
	for j in range(dim[1]):
		for i in range(dim[0]):
			f.write("%.7f %.7f %.7f\n"%(float(coords[index][2]),float(coords[index][1]),float(data[index])))
			# f.write("%.7f %.7f %.7f %.7f\n"%(float(i),float(j),float(k),float(data[index])))
			index+=1
		
	f.close()
def saveFile2Dxy(outputFilePath,data,x,y,dim):
	# XYZ=getBXYZ_3D(dim)
	# _,x,y,z,_=coords.shape
	f=open(outputFilePath,"w")
	
	f.write(" variables=\"x\",\"y\",\"u\" \n" )
	f.write("zone i= "+ '{:4d}'.format(dim[0])+" ,j= "+'{:4d}\n'.format(dim[1]))
	index=0
	for j in range(dim[1]):
		for i in range(dim[0]):
			f.write("%.7f %.7f %.7f\n"%(float(x[index]),float(y[index]),float(data[index])))
			# f.write("%.7f %.7f %.7f %.7f\n"%(float(i),float(j),float(k),float(data[index])))
			index+=1
		
	f.close()
def getDeltaData1(dataPath,dim):
	value=np.zeros([4,(dim[0]),(dim[1])])
	index=0
	with open(dataPath,'r') as f:
		for j in range(dim[1]):
			for i in range(dim[0]):
				line=f.readline().strip().split()
				if math.isnan(float(line[0])):
					print(dataPath," ",i," ",j)
				value[0,i,j]=float(line[0])
				value[1,i,j]=float(line[1])
				value[2,i,j]=float(line[2])
				value[3,i,j]=float(line[3])
				index+=1
	return value
def getCoor5Point(dim,deltad):
    valuet=np.ones((dim[0]*dim[1],5,1))
    valuex=np.zeros((dim[0]*dim[1],5,1))
    valuey=np.zeros((dim[0]*dim[1],5,1))
    valueu=np.zeros((dim[0]*dim[1],5,1))
    index = 0
    
    xinter=0.25/dim[0]
    yinter=1.0/dim[1]
    for j in range(dim[1]):
        for i in range(dim[0]):
            valuex[index,0,0]=i*xinter
            valuex[index,1,0]=(i-1)*xinter
            valuex[index,2,0]=(i-1)*xinter
            valuex[index,3,0]=(i+1)*xinter
            valuex[index,4,0]=(i+1)*xinter
            
            valuey[index,0,0]=j*yinter
            valuey[index,1,0]=(j-1)*yinter
            valuey[index,2,0]=(j+1)*yinter
            valuey[index,3,0]=(j-1)*yinter
            valuey[index,4,0]=(j+1)*yinter
            
            if dim[0]-1>i>0 and dim[1]-1>j>0:
            
                valueu[index,0,0]=deltad[j*dim[0]+i]
                valueu[index,1,0]=deltad[(j-1)*dim[0]+(i-1)]
                valueu[index,2,0]=deltad[(j+1)*dim[0]+(i-1)]
                valueu[index,3,0]=deltad[(j-1)*dim[0]+(i+1)]
                valueu[index,4,0]=deltad[(j+1)*dim[0]+(i+1)]
            else:
                valueu[index,0,0]=deltad[j*dim[0]+i]
                valueu[index,1,0]=deltad[j*dim[0]+i]
                valueu[index,2,0]=deltad[j*dim[0]+i]
                valueu[index,3,0]=deltad[j*dim[0]+i]
                valueu[index,4,0]=deltad[j*dim[0]+i]
            index+=1
    return valuet,valuex,valuey,valueu
def getBData_2D(dataPath,dim):
	value=np.zeros([4,(dim[0]),(dim[1])])
	index=0
	res=open(dataPath,'rb')
	for j in range(dim[1]+1):
		for i in range(dim[0]+1):
			_=res.read(4)
			n1=res.read(4)
			n2=res.read(4)
			n3=res.read(4)
			n4=res.read(4)
			_=res.read(4)
			if i!=dim[0] and j !=dim[1]:
				v1=struct.unpack('1f',n1)
				v2=struct.unpack('1f',n2)
				v3=struct.unpack('1f',n3)
				v4=struct.unpack('1f',n4)
				
				value[0,i,j]=float(v1[0])
				value[1,i,j]=float(v2[0])
				value[2,i,j]=float(v3[0])
				value[3,i,j]=float(v4[0])
				# print(i," ",value[0][index]," ",value[1][index]," ",value[2][index]," ",value[3][index])
			
	res.close()
	return value
def getDeltaData(dataPath,dim):
	value=np.zeros([(dim[0])*(dim[1])])
	index=0
	with open(dataPath,'r') as f:
		for j in range(dim[1]):
			for i in range(dim[0]):
				line=f.readline().strip().split()
				if math.isnan(float(line[0])):
					print(dataPath," ",i," ",j)
				value[index]=float(line[0])
				index+=1
	return value
def get_txy_Vordata(dataPath,dim):
	value=np.zeros([dim,4])
	# index=0
	with open(dataPath,'r') as f:
		for i in range(dim):
			line=f.readline().strip().split()
			# if math.isnan(float(line[0])):
				# print(dataPath," ",i," ",j)
			value[i,0]=float(line[0])
			value[i,1]=float(line[1])
			value[i,2]=float(line[2])
			value[i,3]=float(line[3])
	return value
	
def get_txy_Vordata5Point(dataPath):
	# index=0
	with open(dataPath,'r') as f:
		line=f.readline().strip().split()
		dim=int(line[0])
		# print(dim)
		value=np.zeros([dim//10,5,4,1])
		for i in range(dim//10):
			for j in range(5):
				line=f.readline().strip().split()
				value[i,j,0,0]=float(line[0])
				value[i,j,1,0]=float(line[1])
				value[i,j,2,0]=float(line[2])
				value[i,j,3,0]=float(line[3])
			for _ in range(5):
				line=f.readline().strip().split()
	return value

class RT3_16X(Dataset) : #héritage de classe Dataset de Pytorch
	def __init__(self,start=0, end=2000, train=False):
		super().__init__()
		self.mode = 'train'*train + 'val'*(1-train)
		
		if train:
			datapath='/home/nvme1/slm/sr/SamplingDeltaData_SameCoordContinue9Point32000/RT35_16X/'#12800_Random12800
			dataU = []
			datax = []
			datay = []
			datat = []
			
			print("load data")
			for i in range(start,end):
				data=get_txy_Vordata5Point(datapath+'{:04d}RT35_16X_delta_Cluster_01.dat'.format(i))
				datat += list(data[:,:,0,:])
				datax += list(data[:,:,1,:])
				datay += list(data[:,:,2,:])
				dataU += list(data[:,:,3,:])
				
			dataU = np.asarray(dataU)
			print(np.min(dataU)," ",np.max(dataU))
			dataU = 2*(dataU -(-1.2))/(1.2-(-1.2))-1
			# dataU = dataU.reshape(-1,-1, 1)
			datat = np.asarray(datat)
			datax = np.asarray(datax)
			datay = np.asarray(datay)
			# datat = datat.reshape(-1,-1, 1)
			# datax = datax.reshape(-1,-1, 1)
			# datay = datay.reshape(-1, -1,1)
			# print(datax.shape)
			self.datat = torch.tensor(datat).float()
			self.datax = torch.tensor(datax).float()#.view(size) permet de modifier la shape et d'utiliser le même espace de stockage
			self.datay = torch.tensor(datay).float() #labels
			self.dataU = torch.tensor(dataU).float()
			# print(self.datax)
		else:
			data_pathHigh ='/home/nvme0/dx/dl/weno_code/weno5/RT1920/' #'../../../../nvme0/dx/dl/weno_code/weno3/DM1920/'
			dataU = []
			datax = []
			datay = []
			datat = []

			print("load data")
			deltad = getDeltaData1(data_pathHigh+'35Deltafull'+'{:04d}'.format(start)+'.DAT',[480,1920])[0]
			deltad = deltad.flatten('F')
			dataCoor5Point = getCoor5Point([480,1920],deltad)#= datacoord 
			datat += list(dataCoor5Point[0])
			datax += list(dataCoor5Point[2])
			datay += list(dataCoor5Point[1])
			dataU += list(dataCoor5Point[3])
	
			dataU = np.asarray(dataU)
			print(np.min(dataU)," ",np.max(dataU))
			dataU = 2*(dataU-(-1.2))/(1.2-(-1.2))-1
			datat = np.asarray(datat)
			datax = np.asarray(datax)
			datay = np.asarray(datay)
			# datat = datat.reshape(-1,-1, 1)
			# datax = datax.reshape(-1,-1, 1)
			# datay = datay.reshape(-1, -1,1)
			# print(datax.shape)
			self.datat = torch.tensor(datat).float()
			self.datax = torch.tensor(datax).float()#.view(size) permet de modifier la shape et d'utiliser le même espace de stockage
			self.datay = torch.tensor(datay).float() #labels
			self.dataU = torch.tensor(dataU).float()
			# print(self.datax)
		print("load end")
	
	def __getitem__(self ,index):
		"""retourne un couple (exemple,label) correspondant à l’index"""
	
		return self.datat[index, :], self.datax[index,:],self.datay[index, :], self.dataU[index,:]
		
	def __len__(self):
		"""renvoie la taille du jeu de donnees"""
		return self.datax.shape[0]
