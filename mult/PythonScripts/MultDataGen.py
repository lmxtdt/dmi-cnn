#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 12:47:15 2022

@author: liathomson

New Generator, for 6 chromosomes of 1,000 bp each
"""
import numpy as np
from glob import glob
import tensorflow as tf

class ChrGen(tf.keras.utils.Sequence):
	def __init__(self, npzGlob, batchSize = 60, shuffle = True, 
			  excludeStats = True):
		self.batchSize = batchSize
		self.shuffle = shuffle
		self.excludeStats = excludeStats
		
		self.npzFiles = glob(npzGlob)
		
		#load first file
		self.currFile = -1
		self.currX = None
		self.currY = None
		
		self.currChiP = None
		
		self.updateFile(0, firstFile = True)
	
	def __len__(self):
		return self.batchesPerFile * len(self.npzFiles)
	
	def adjustCurr(self):
		#y is a (SAMPLENUM, 1000 )array
		#first, reduce y to a (SAMPLENUM, 100) array
		#each 10-number-long bin will be represented
		#by 1 number, equal to the max value of the bin
		#shrinking each 10-locus long bin into a single
				
		binned = np.reshape(self.currY, (-1, 100, 10))
		shrunk = np.max(binned, axis = 2)
				
		#then, increase the values of the locations around
		#peaks in y
		f = 1.5
		
		#newY is updated to contain the desired values
		newY = np.copy(shrunk)
		#oldY stores the values of the previous iteration, with
		#padding
		oldY = np.zeros((newY.shape[0], newY.shape[1] + 2), dtype = float)
		for i in range(15):
			#update oldY
			oldY[:,1:-1] = newY
			
			#for each sample
			for s in range(newY.shape[0]):
				#for each location (in newY)
				for j in range(newY.shape[1]):
					#use the values from oldY corresponding to the left (s, j), 
					#center (s, j + 1) and right (s, j + 2) of newY[s,j]
					newY[s,j] = max(oldY[s, j] / f,
									    oldY[s, j + 1], 
										oldY[s, j + 2] / f)
		
		self.currY = newY
		
		#update currChiP to be the p-value from chi squared analysis
		self.currChiP = self.currX[:,:,6]
		
		#adjust currX to exclude the statistical numbers
		#(indices 4, 5, and 6 from axis 3)
		if(self.excludeStats):
			self.currX = self.currX[:, :, 0:4]
		
	def updateInternal(self):
		self.samplesPerFile = self.currY.shape[0]

		if(self.samplesPerFile % self.batchSize != 0):
			raise Exception("batchSize is {},"
							" must be a factor of {}".format(self.batchSize, 
															 self.samplesPerFile))
		else:
			self.batchesPerFile = self.samplesPerFile // self.batchSize
		
	def updateFile(self, fileIdx, firstFile = False):
		self.currFile = fileIdx
		npzFile = np.load(self.npzFiles[fileIdx])
		
		self.currX = npzFile.get("s")
		self.currY = npzFile.get("y")
		
		npzFile.close()
		
		if(firstFile):
			#update internal stats
			self.updateInternal()
		
		self.adjustCurr()
		
		#shuffle order
		indices = np.arange(0, len(self.currY))
		np.random.shuffle(indices)
		
		self.currX = tf.gather(self.currX, indices)
		self.currY = tf.gather(self.currY, indices)
		self.currChiP = tf.gather(self.currChiP, indices)
	
	def __getitem__(self, index):
		fileIdx = index // self.batchesPerFile
		batchIdx = index % self.batchesPerFile
		
		#open correct file
		if(fileIdx != self.currFile):
			self.updateFile(fileIdx)

		#get currect slices
		start = batchIdx * self.batchSize
		end = (batchIdx + 1) * self.batchSize
		batchX = self.currX[start : end]
		batchY = self.currY[start : end]
		
		return batchX, batchY
	
	def getXYChi(self, index):
		fileIdx = index // self.batchesPerFile
		batchIdx = index % self.batchesPerFile
		
		#open correct file
		if(fileIdx != self.currFile):
			self.updateFile(fileIdx)

		#get currect slices
		start = batchIdx * self.batchSize
		end = (batchIdx + 1) * self.batchSize
		batchX = self.currX[start : end]
		batchY = self.currY[start : end]
		batchChi = self.currChiP[start : end]
		
		return batchX, batchY, batchChi
		
	def on_epoch_end(self):
		#shuffle the order of the files of each category
		if(self.shuffle):
			np.random.shuffle(self.npzFiles)
			self.currFile = -1
			self.currX = None
			self.currY = None


class FiltChrGen(ChrGen):
        #different from ChrGen because it sorts samples
        #and picks the most severe
        #(tried using it, not really worth it)
	def adjustCurr(self):
		
		#halve currX and currY
		degree = np.max(self.currY, axis = 1)
		order = np.argsort(degree)
		select = order[int(len(order) / 2) :]
		
		self.currY = tf.gather(self.currY, select)
		self.currX = tf.gather(self.currX, select)
		
		
		#y is a (SAMPLENUM, 1000 )array
		#first, reduce y to a (SAMPLENUM, 100) array
		#each 10-number-long bin will be represented
		#by 1 number, equal to the max value of the bin
		#shrinking each 10-locus long bin into a single
				
		binned = np.reshape(self.currY, (-1, 100, 10))
		shrunk = np.max(binned, axis = 2)
				
		#then, increase the values of the locations around
		#peaks in y
		f = 1.5
		
		#newY is updated to contain the desired values
		newY = np.copy(shrunk)
		#oldY stores the values of the previous iteration, with
		#padding
		oldY = np.zeros((newY.shape[0], newY.shape[1] + 2), dtype = float)
		for i in range(15):
			#update oldY
			oldY[:,1:-1] = newY
			
			#for each sample
			for s in range(newY.shape[0]):
				#for each location (in newY)
				for j in range(newY.shape[1]):
					#use the values from oldY corresponding to the left (s, j), 
					#center (s, j + 1) and right (s, j + 2) of newY[s,j]
					newY[s,j] = max(oldY[s, j] / f,
									    oldY[s, j + 1], 
										oldY[s, j + 2] / f)
		
		self.currY = newY
		
		#adjust currX to exclude the statistical numbers
		#(indices 4, 5, and 6 from axis 3)
		if(self.excludeStats):
			self.currX = self.currX[:, :, 0:4]
		
			
	def updateInternal(self):
		if(self.currY.shape[0] % 2):
			raise Exception("file must contain even number of samples")
		self.samplesPerFile = int(self.currY.shape[0] / 2)

		if(self.samplesPerFile % self.batchSize != 0):
			raise Exception("batchSize is {},"
							" must be a factor of {}".format(self.batchSize, 
															 self.samplesPerFile))
		else:
			self.batchesPerFile = self.samplesPerFile // self.batchSize
