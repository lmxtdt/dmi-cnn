#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 19:02:38 2022

@author: liathomson
"""
#import statements
from sys import argv
import tensorflow as tf
import numpy as np
from MultDataGen import ChrGen
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

#get parameters
modelPath = argv[1]
dataGlob = argv[2]
excludeStats = bool(int(argv[3]))

#another function to find the peaks in output
#not actually used; find_peaks from scipy is the one used
def find_peaks2(arr, height = None, flank = 1):
	if(height == None):
		height = np.min(arr)
				
	a = np.pad(arr, pad_width = flank, mode = "minimum")
	#finds peaks, defined as indices for which
	#the value is greater than the values of all of its neighbors
		
	peaks = []
	for i in range(flank, a.shape[0] - flank):
		curr = a[i]
		
		if(curr > height):
			larger = []
			for j in range(i - flank, i + flank + 1):
				larger.append(curr >= a[j])
						
			if(all(larger)):
				peaks.append(i - flank)
								
	filtPeaks = []
	
	i = 0
	while(i < len(peaks)):
		if((i < (len(peaks) - 1)) and (peaks[i] == (peaks[i + 1] - 1))):
			start = peaks[i]
			while((i < (len(peaks) - 1)) and (peaks[i] == (peaks[i + 1] - 1))):
				i += 1
			end = peaks[i]
			
			filtPeaks.append(np.mean([start, end], dtype = int))
		else:
			filtPeaks.append(peaks[i])
		i += 1
	
	return np.array(filtPeaks)

#class that saves different metrics for evaluating accuracy
class peakMetrics:
	def __init__(self, width):
		self.width = width #5 cM or 10 cM
		
		self.realFound = [] #real peaks that were predicted, % by chromosome
		self.realDist = []  #distance between real peak and closest predicted
							#assuming it was predicted
		self.realNum = []   #mean # predicted peaks per real peak
		self.predCorrect = [] #number predicted peaks that found a real one
		self.predReal = []  #number predicted peaks - number real peaks
		self.predRealDiv = [] #number pred - num real / num pred + num real
	
	#for sorting
	def __gt__(self, other):
		return np.mean(self.predRealDiv) > np.mean(other.predRealDiv)
	
	def __lt__(self, other):
		return np.mean(self.predRealDiv) < np.mean(other.predRealDiv)
	
	#returns number of simulations evaluated
	def getNumSims(self):
		return len(self.predReal)
	
	def getMean(self):
		#mean everything
		d = {"realFound": np.mean(self.realFound),
			   "realDist": np.mean(self.realDist),
			   "realNum": np.mean(self.realNum),
			   "predCorrect": np.mean(self.predCorrect),
			   "predReal": np.mean(self.predReal),
			   "predRealDiv": np.mean(self.predRealDiv)}
		
		return d
	
	#update self using real peaks and predicted peaks
	def updateMetrics(self, r, p):
		found = []
		closestDist = []
		correctPeaks = []
				
		for rPeak in r:
			#check if any peaks were found
			if(p.shape[0] == 0):
				found.append(False)
			else:
				#get distances between real peak & predicted
				distances = np.abs(rPeak - p)
				#filter distances to only be those within the given width
				foundDist = distances[distances <= self.width]
				foundPeaks = p[distances <= self.width]
				
				nFound = foundDist.shape[0]
				
				#record whether this peak was found
				if(nFound):
					found.append(True)
				
					#find the closest peak, add its distance to closestDist
					closestDist.append(np.min(foundDist))
				
					#record the found peaks
					correctPeaks.extend(foundPeaks)
					
				else:
					found.append(False)
			
		#print("r: {}\np:{}\nfound: {}\nclosestDist: {}\ncorrectPeaks: {}\n".format(
	#		r, p, found, closestDist, correctPeaks))
		
		#get mean % real peaks found and mean dist. between real peak and pred.
		if(r.shape[0] > 0):
			self.realFound.append(np.mean(found))
			
		if(closestDist != []):
			self.realDist.append(np.mean(closestDist))
		
			#add to realNum, the mean # predicted peaks per found real peak
			self.realNum.append(len(correctPeaks) / np.sum(found))
			#add to predCorrect, the mean # predicted peaks close to a real peak
			self.predCorrect.append(np.unique(correctPeaks).shape[0] / p.shape[0])
		
		#add num. pred peaks - num. real peaks
		self.predReal.append(p.shape[0] - r.shape[0])
		self.predRealDiv.append((p.shape[0] - r.shape[0]) / (p.shape[0] + r.shape[0] + 1e-5))
		
#class that evaluates different methods of finding incompatibilities
#each instance is one method (e.g. CNN vs Chi-squared)
#contains a peakMetrics instance
class paramSet:
	def __init__(self, typ, smooth, thresh, width):
		self.typ = typ #chi or cnn
		self.smooth = smooth #true or false
		self.thresh = thresh #threshold for the value
		self.width = width   #minimum distance between loci to be considered distinct

		self.metrics = peakMetrics(width)
		
	def __gt__(self, other):
		return self.metrics > other.metrics
	
	def __lt__(self, other):
		return self.metrics < other.metrics
		
	def getMean(self):
		return self.metrics.getMean()
	
	def getMetrics(self):
		return self.metrics
		
	#update metrics using the y output from the data
	#and either the prediction from the CNN or the chi-squared value
	def updateMetrics(self, y, predChi):
		#smooth if necessary
		if(self.smooth):
			predChi = gaussian_filter(predChi, 0.5)
		#invert predChi if it's chi-squared p-value
		if(self.typ == "chi"):
			predChi = predChi * -1
		
		#find peaks
		p = find_peaks(predChi, height = self.thresh)[0]
			
		r = find_peaks(y, height = 0.05)[0]
				
		#update metrics by passing in the peaks
		self.metrics.updateMetrics(r, p)

	def __str__(self):
		return "<paramSet. {}. smooth: {}. thresh: {}. width: {}.>".format(
																self.typ,
																self.smooth, 
																self.thresh,
																self.width)

#the class that contains all paramSets to be examined
class testSet:
	def __init__(self):
		self.numTested = 0
		self.chiSets = []
		self.cnnSets = []

		#fill in the desired paramSets
		for width in [5, 10, 20]:
			for smooth in [True, False]:
				for thresh in [0.05, 0.075, 0.1]:
					self.cnnSets.append(paramSet("cnn", smooth, thresh, width))
				for thresh in [-1e-5, -1e-10, -1e-15]:
					self.chiSets.append(paramSet("chi", smooth, thresh, width))
		
	#given y, pred from CNN, and chi-squared values
	#update all internal metrics
	def updateMetrics(self, y, pred, chi):
		self.numTested += 1
		
		for chiSet in self.chiSets:
			chiSet.updateMetrics(y, chi)
		
		for cnnSet in self.cnnSets:
			cnnSet.updateMetrics(y, pred)
	
	def sortSets(self):
		self.chiSets = list(np.sort(self.chiSets))
		self.cnnSets = list(np.sort(self.cnnSets))
		
	def __str__(self):
		return "<testSet. tested against {}>".format(self.numTested)
	
	def printMetrics(self):
		self.sortSets()
		#first print out chiSets
		print("chi squared parameters:")
		print("-----------------------")
		for cs in self.chiSets:
			print(cs)
			d = cs.getMean()
			for key in d.keys():
				print("\t{}: {}".format(key, d[key]))
			print("\n-----")
	
		print("CNN prediction parameters:")
		print("--------------------------")
		for cs in self.cnnSets:
			print(cs)
			d = cs.getMean()
			for key in d.keys():
				print("\t{}: {}".format(key, d[key]))
			print("\n-----")
		
	
#now, set up the model.
#because this is a poorly organized file

#define IOU functions so models that use IOU will load properly
def calcIOU(real, pred):
	intersection = tf.minimum(real, pred)
	union = tf.maximum(real, pred)
	unionCorr = tf.maximum(union, 1e-5) #correct to avoid dividing by 0
	
	iou = tf.divide(intersection, unionCorr)
	return tf.subtract(1.0, iou)
	
def sumIOU(real, pred):
	minusIOU = calcIOU(real, pred)
	return tf.reduce_sum(minusIOU)
	
def meanIOU(real, pred):
	minusIOU = calcIOU(real, pred)
	return tf.reduce_mean(minusIOU)

def meanChi(chi):
	#assumes chi is formatted to size (variable, 1000)
	#want to return something of size (variable, 100)
	#do so by returning the minimum chi (p-value) for each 10-size bin
	binned = tf.reshape(chi, (-1, 100, 10))
	meanChi = tf.reduce_mean(binned, axis = 2)
	
	return meanChi

#load the model
model = tf.keras.models.load_model(modelPath, custom_objects = {"meanIOU": meanIOU, 
																"sumIOU": sumIOU})

#load in test data
testGen = ChrGen(dataGlob, batchSize = 24, excludeStats = excludeStats)

#create testSet
ts = testSet()

#go through all of the test data
for i in range(len(testGen)):
	#first, get x
	x, y, chi = testGen.getXYChi(i)
	
	#reshape chi p-values into 100 long, using the mean of each 10 bp bin
	reshapedChi = meanChi(chi)
	
	#get model's predictions from x
	p = model.predict(x)
	
	#print("x: {}, y: {}, chi: {}, p: {}".format(type(x), type(y), type(reshapedChi), type(p)))
	
	#compare peaks for each sample in the batch
	for b in range(p.shape[0]):
		ts.updateMetrics(y[b].numpy(), p[b], reshapedChi[b].numpy())

#print out metrics for all examined parameter sets
ts.printMetrics()