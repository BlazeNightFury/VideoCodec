# Just run this program with the image and Huffman.py file in same folder as this code


import time
import math
import heapq
import cv2
from cv2.cv import *
import numpy as np
import numpy
from matplotlib import pyplot as plt
from Huffman_for_hw2 import *

array = cv2.imread('aerial.tiff',0)

Q = [[16.0,11.0,10.0,16.0,24.0,40.0,51.0,61.0],
	[12.0,12.0,14.0,19.0,26.00,58.0,60.0,55.0],
	[14.0,13.0,16.0,24.0,40.0,57.0,69.0,56.0],
	[14.0,17.0,22.0,29.0,51.0,87.0,80.0,62.0],
	[18.0,22.0,37.0,56.0,68.0,109.0,103.0,77.0],
	[24.0,35.0,55.0,64.0,81.0,104.0,113.0,92.0],
	[49.0,64.0,78.0,87.0,103.0,121.0,120.0,101.0],
	[72.0,92.0,95.0,98.0,112.0,100.0,103.0,99.0]]

def Encoder(array,N,M):
	#Encoding
	ind = 0
	blocks = np.ndarray(shape=(array.size/(N*M),N,M))
	Qblocks = np.ndarray(shape = (array.size/(N*M),N*M))
	for i in xrange(0,array.shape[0],N):
		for j in xrange(0,array.shape[1],M):
			# 'Making image into blocks'
			blocks[ind,:,:] = array[i:i+N,j:j+M]
			# 'Applying DCT'
			blocks[ind,:,:] = DCT(blocks[ind,:,:],N,M)
			#'Quantising'
			blocks[ind,:,:] = np.floor(np.divide(blocks[ind,:,:],Q)+0.5)
			# 'Rearraning'
			Qblocks[ind,:],zzind = zigzag(blocks[ind,:,:])
			ind+=1
	DC = np.ndarray(shape = (Qblocks.shape[0]))
	DC = Qblocks[:,0]
	AC = np.ndarray(shape = (Qblocks.shape[0],Qblocks.shape[1]-1))	
	AC = Qblocks[:,1:Qblocks.shape[1]]
	# 'Forming AC and DC coeeficients'
	dpcmDC = DPCM(DC)
	rlcAC = RLC(AC)
	rlcAC,HuffDict = rlcHuff(rlcAC) #huffman encode for rlc
		
	return dpcmDC,rlcAC,HuffDict,zzind

def Decoder(dpcmDC,rlcAC,HuffDict,zzind,N,M,shape):
	#Decoding
	rlcAC = derlcHuff(rlcAC,HuffDict)	#huffman decode for rlc
	# 'Decoding AC and DC coeeficients'
	AC = deRLC(rlcAC)
	DC = deDPCM(dpcmDC)
	reconArray = np.ndarray(shape = (shape))
	blocks = np.ndarray(shape=(DC.shape[0],N,M))
	Qblocks = np.ndarray(shape = (AC.shape[0],AC.shape[1]+1))
	Qblocks[:,0] = DC
	Qblocks[:,1:Qblocks.shape[1]] = AC
	for ind in xrange(0,DC.shape[0]):
		for i in xrange(0,len(zzind)):
			# 'Derranging'
			blocks[ind,zzind[i][0],zzind[i][1]] = Qblocks[ind,i]
	ind = 0
	for i in xrange(0,shape[0],N):
		for j in xrange(0,shape[1],M):
			# 'Requantising'
			blocks[ind,:,:] = np.ceil(np.multiply((blocks[ind,:,:]),Q))
			# 'Applying IDCT'
			# 'Reconstructing array'
			reconArray[i:i+N,j:j+M] = IDCT(blocks[ind,:,:],N,M)
			ind+=1
	return reconArray

def DCT(blocks,N,M):
	I = np.ndarray(shape=(N,M))
	for u in xrange(0,N):
		for v in xrange(0,M):
			cosiu = np.cos(np.multiply((((np.multiply(2.0,range(0,N)))+([1.0]*N))*(np.pi))/(2.0*N),u))
			cosjv = np.cos(np.multiply((((np.multiply(2.0,range(0,M)))+([1.0]*M))*(np.pi))/(2.0*M),v))
			if u==0 or v==0:
				const = np.sqrt(2.0)/(N*M)
			if v==0 and u==0:
				const = 1.0/(N*M)
			else :
				const = 2.0/(N*M)
			I[u,v] = (np.dot(np.dot(cosiu,blocks),cosjv))*const
	return I

def IDCT(blocks,N,M):
	I = np.ndarray(shape=(N,M))
	for i in xrange(0,N):
		for j in xrange(0,M):			
			cosiu = np.cos(np.multiply((((2.0*i)+1.0)*(np.pi/(2.0*N))),range(0,N)))
			cosjv = np.cos(np.multiply((((2.0*j)+1.0)*(np.pi/(2.0*M))),range(0,M)))
			cn = np.ones(N)
			cm = np.ones(M)
			cn[0]=1.0/np.sqrt(2.0)
			cm[0]=1.0/np.sqrt(2.0)
			cosiu = np.multiply(cosiu,cn)
			cosjv = np.multiply(cosjv,cm)
			I[i,j] = (np.dot(np.dot(cosiu,blocks),cosjv))*2.0
	return I

def zigzag(I):
	zz = [] 
	zzind = []
	j = i = 0
	for s in xrange(0,16):
		if s/2.0 != 0:
			i = 0
			j = s
			while j>=0:
				if j<8 and i<8:
					zz.append(I[i,j])
					zzind.append([i,j])
				j-=1
				i+=1
		elif s/2.0 == 0:
			j = 0
			i = s
			while i>=0:
				if i<8 and j<8:
					zz.append(I[i,j])
					zzind.append([i,j])
				i-=1
				j+=1
	return zz,zzind

def DPCM(DC):
	dpcmDC = np.ndarray(shape = (DC.shape[0],))
	dpcmDC[0] = DC[0]
	for i in xrange(1,DC.shape[0]):
		dpcmDC[i] = DC[i]-DC[i-1]
	return dpcmDC 

def RLC(AC):
	rlcAC = []
	for l in AC:
		net = 0
		freq = 1
		temp = []
		for i in xrange(1,len(l)):
			if l[i]!= l[i-1]:
				temp.append((freq,l[i-1]))
				net+=freq
				if i==(len(l)-1):
					freq = 1
					temp.append((freq,l[i]))
				freq = 1
			elif l[i]==l[i-1]:
				if i==(len(l)-1):
					freq+=1
					temp.append((freq,l[i]))
					net+=1
				freq+=1
		rlcAC.append(temp)
		del temp
	return rlcAC

def deRLC(rlcAC):
	AC = []
	for l in rlcAC:
		temp = []
		for freq,num in l:
			temp+=[num]*freq
		AC.append(temp)
		del temp
	AC = np.array(AC)
	return AC

def deDPCM(dpcmDC):
	DC = np.ndarray(shape = (dpcmDC.shape[0],))
	DC[0] = dpcmDC[0]
	for i in xrange(1,dpcmDC.shape[0]):
		dpcmDC[i] = dpcmDC[i]+dpcmDC[i-1]
		DC[i] = dpcmDC[i]
	return DC

def rlcHuff(rlcAC):
	rlcRavel = []
	for l in rlcAC:
		rlcRavel =  rlcRavel + l
	rlcRavel = np.array(rlcRavel)
	syms = rlcRavel[:,1]
	freqs = rlcRavel[:,0]
	u = np.unique(syms)
	netFreq = []
	for ele in u:
		netFreq.append(sum(freqs[syms==ele]))
	netFreq = np.array(netFreq)
	probs = (netFreq/float(sum(netFreq))).tolist()
	n = HuffmanTree(u,probs)
	HuffDict = HuffmanDict(n)
	for l in rlcAC:
		for freq,num in l:
			num = HuffDict[num]
	return rlcAC,HuffDict

def derlcHuff(rlcAC,HuffDict):
	sym = HuffDict.keys()
	code = HuffDict.values()
	for l in rlcAC:
		for freq,num in l:
			num = sym[code==num]
	return rlcAC


dpcmDC,rlcAC,HuffDict,zzind = Encoder(array,8,8)
reconArray = Decoder(dpcmDC,rlcAC,HuffDict,zzind,8,8,array.shape)
plt.subplot(1,2,1)
plt.imshow(array,'gray')
plt.xticks([]), plt.yticks([])
plt.title("Actual Image")
plt.subplot(1,2,2)
plt.imshow(reconArray,'gray')
plt.xticks([]), plt.yticks([])
plt.title("Reconstructed Image")
plt.show()
