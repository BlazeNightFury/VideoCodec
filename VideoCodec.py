# Just run this program with frame1.jpg,frame2.jpg and this code in the same folder 

import time
import math
import heapq
import cv2
from cv2.cv import *
import numpy as np
import numpy
import copy
from matplotlib import pyplot as plt

def MotionEstimation(curr,buff,N,M):
	#Blocks
	ind = 0
	ME = []
	centers = []
	MC = np.ndarray(shape=curr.shape)
	print("Dividing into macroblocks...")
	for i in range(0,curr.shape[0],N):
		for j in range(0,curr.shape[1],M):
			# 'Making current frame into blocks of dimension NxM and performing three step search'
			ME = ME + TSS(curr[i:i+N,j:j+M],buff,np.floor((i+i+N)/2),np.floor((j+j+M)/2)) # Finding motion vectors by motion estimation
			centers = centers + [[np.floor((i+i+N)/2),np.floor((j+j+M)/2)]] # Centres of eac macroblock
			MC[i:i+N,j:j+M] = buff[ME[ind][0]-(N/2):ME[ind][0]+(N/2),ME[ind][1]-(M/2):ME[ind][1]+(M/2)] # Motion Compensated blocks
			ind+=1
	return MC,ME,centers

def MAD(pind,curr,buff,N,M):
	#Mean Absolute Deviation
	mad = []
	buffblock = np.zeros(curr.shape)
	pnew = []
	for p in pind:
		#'Blocks of the frame in buffer'
		buffblock = buff[p[0]-(N/2):p[0]+(N/2),p[1]-(M/2):p[1]+(M/2)]

		if buffblock.shape == curr.shape:
			mad = mad + [sum(sum(abs(buffblock.astype(float)-curr.astype(float))))/float(N*M)] # Computing MAD
			pnew = pnew + [p]

	return mad,pnew

def TSS(currblock,buff,i,j):
	#Three Step Search
	step = 4
	while step>0:
		pind = points(buff,i,j,step)
		bm,pnew = MAD(pind,currblock,buff,N,M)
		bm = bm.index(min(bm))
		i,j = pnew[bm]
		step = step/2
	ME = [[i,j]]
	return ME

def points(arr,i,j,step):
	# Selecting the 9 points around i,j(center) with the given step size
	pind = [0]*9
	pind[0] = [i,j]
	pind[1] = [i-step,j-step]
	pind[2] = [i,j-step]
	pind[3] = [i-step,j]
	pind[4] = [i+step,j+step]
	pind[5] = [i,j+step]
	pind[6] = [i+step,j]
	pind[7] = [i-step,j+step]
	pind[8] = [i+step,j-step]

	return pind


print("Reading the fisrt two Frames...")

frame1 = cv2.imread('frame1.jpg',0)
frame2 = cv2.imread('frame2.jpg',0)

N=16
M=16
MC,ME,centers = MotionEstimation(frame2,frame1,N,M)
print("Computed the motion vectors and motion compensation...")

ME = np.array(ME)
centers = np.array(centers)
img = np.zeros(MC.shape,np.uint8)
temp_MC = copy.deepcopy(MC)

#plotting Motion Vectors
for i in range(len(ME)):
	cv2.line(img,(int(ME[i,1]),int(ME[i,0])),(int(centers[i,1]),int(centers[i,0])),1,1,4)
	cv2.line(temp_MC,(int(ME[i,1]),int(ME[i,0])),(int(centers[i,1]),int(centers[i,0])),1,1,4)

print("Plotting ...")

plt.figure()
plt.suptitle("Motion Vectors")

plt.subplot(1,2,1)
plt.imshow(img,'gray')
plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(temp_MC,'gray')
plt.xticks([]), plt.yticks([])

plt.figure()
plt.subplot(2,2,1)
plt.imshow(frame1,'gray')
plt.xticks([]), plt.yticks([])
plt.title("Frame in Buffer")

plt.subplot(2,2,2)
plt.imshow(frame2,'gray')
plt.xticks([]), plt.yticks([])
plt.title("Current Frame")

plt.subplot(2,2,3)
plt.imshow(MC,'gray')
plt.xticks([]), plt.yticks([])
plt.title("Motion Compensated Frame")


plt.subplot(2,2,4)
plt.imshow(abs(MC-frame2),'gray')
plt.xticks([]), plt.yticks([])
plt.title("Abs Error with motion compensation")

plt.show()

print("End of Program!")
