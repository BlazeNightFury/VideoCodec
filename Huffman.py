import time
import math
import heapq
import cv2
from cv2.cv import *
import numpy as np
import numpy
from matplotlib import pyplot as plt

class Node(object):
	"""docstring for Node"""
	def __init__(self,mom,dad,child,code,sym,prob):
		self.mom = mom
		self.dad = dad
		self.child = child
		self.code = code
		self.sym = sym
		self.prob = prob
		self.visit = 0

def find(list1,list2,ele):
	flag = 0
	found = Node(None,None,None,'',0,0)
	for i in range(0,len(list2)):
		if list2[i]!= None :
			if list2[i].visit==0:
				if list2[i].prob==ele:
					found = list2[i]
					list2[i].visit = 1
					flag = 1
					break
	if flag == 0:
		for i in range(0,len(list1)):
			if list1[i].visit==0:
				if list1[i].prob==ele:
					found = list1[i]
					list1[i].visit = 1
					flag = 1
					break
	return found

def HuffmanTree(sym,probs):
	len_probs = len(probs)
	n = [None] * len_probs
	for i in xrange(0,len_probs):
		n[i] = Node(None,None,None,'',sym[i],probs[i])
	heapq.heapify(probs)
	new_n = [None]*(len_probs-1)

	for i in xrange(0,len_probs-1):
		# print(i)
		# print(probs)
		min1 = heapq.heappop(probs)
		min2 = heapq.heappop(probs)
		# print('min1',min1)
		# print('min2',min2)
		node_min1 = Node(None,None,None,'',None,0.0)
		node_min2 = Node(None,None,None,'',None,0.0)
		node_min1 = find(n,new_n,min1)
		node_min2 = find(n,new_n,min2)
		new = min1 + min2
		new_n[i] = Node(None,None,None,'',None,new)
		new_n[i].mom = node_min1
		new_n[i].dad = node_min2
		node_min1.child = new_n[i]
		node_min2.child = new_n[i]
		heapq.heappush(probs,new)
		# print(new_n[i].prob)
		# print('-------------------')

	j = len_probs-2
	while j>-1:
		# print(j)
		# print(new_n[j].prob)
		# print(new_n[j].mom.prob)
		# print(new_n[j].dad.prob)
		new_n[j].mom.code = new_n[j].mom.code + new_n[j].code + '0'
		new_n[j].dad.code = new_n[j].dad.code + new_n[j].code + '1'
		# print(new_n[j].mom.code)
		# print(new_n[j].dad.code)
		# print('-------------------')
		j-=1	
	return n

def HuffmanDict(n):
	HuffDict = {}
	for x in xrange(0,len(n)):
	 	# print(n[x].prob,n[x].code)
	 	temp = {n[x].sym : n[x].code}
	 	HuffDict.update(temp)
	return HuffDict

def HuffmanEncode(sym,probs,inStr):
	n = HuffmanTree(sym,probs)
	HuffDict = HuffmanDict(n)
	# print(HuffDict.values())
	# print(HuffDict.keys())
	encoded = ''
	for ele in inStr:
		encoded = encoded + HuffDict[ele]
	# print(encoded)
	# inStr = np.arr
	return encoded,HuffDict	

def HuffmanDecode(encoded,HuffDict):
	temp = ''
	decoded = ''
	found = []
	i = 0
	for s in encoded:
		if found == []:
			temp = temp + s
		else:
			decoded = decoded + found[0]
			temp = s
			i+=1
		found = [key for key,value in HuffDict.items() if value == temp]
	decoded = decoded + found[0]
	# print(decoded)
	return(decoded)

def avgCL(HuffDict,probs,sym):
	probs = np.array(probs)
	sym = np.array(sym)
	l = [probs[np.nonzero(sym == key)][0]*len(value) for key,value in HuffDict.items()]
	return sum(l)

sym = ['a','e','i','o','u']
probs = [0.25,0.25,0.2,0.15,0.15]
inStr = 'aeiou'

start = time.time()
[encoded,HuffDict] = HuffmanEncode(sym,probs,inStr)
stop = time.time()
#print(stop - start)
decoded = HuffmanDecode(encoded,HuffDict)

probs = [0.25,0.25,0.2,0.15,0.15]
# cl = avgCL(HuffDict,probs,sym)
