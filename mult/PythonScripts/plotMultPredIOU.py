#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import statements
from sys import argv
#from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from MultDataGen import ChrGen
from scipy.stats import chisquare
import csv

#get parameters
modelPath = argv[1]
dataGlob = argv[2]
outFolder = argv[3]
excludeStats = bool(int(argv[4]))
negIndex = 3

#define IOU-related functions
#so if a model is loaded that is meant to use one of them
#it will work
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

#load the model
model = tf.keras.models.load_model(modelPath, custom_objects = {"meanIOU": meanIOU, "sumIOU": sumIOU})

#model.summary()

#saliency map stuff
lastConvLayer = model.layers[-negIndex]

print(lastConvLayer.name)

getMaps = tf.keras.models.Model(inputs = [model.inputs],
								outputs = [model.output, lastConvLayer.output])

#open the npz file
gen = ChrGen(dataGlob, batchSize = 6, excludeStats = excludeStats)
xArr, yArr, chiArr = gen.getXYChi(0)
modelName = modelPath.split("/")[-1]

#look at 6 example chromosomes
for i in range(6):
	#load in y, x, and chi-squared analysis of genotype counts
	y = yArr.numpy()[i]
	loaded = xArr.numpy()[i]
	chi = chiArr[i]
	
	#change dimensions of x so it will work
	x = np.expand_dims(loaded, axis = 0)
	
	#define output names of the figures and the csv
	figName = "{}/{}_{}.png".format(outFolder, modelName, i)
	csvName = "{}/{}_{}.csv".format(outFolder, modelName, i)
	
	#use model to generate predictions
	if(excludeStats):
		#truncate x to exclude statistical features
		#(chi-squared value, chi-squared p-value, multinomial p-value)
		#if the model was trained without those stats
		p = model.predict(x[:,:,0:4])
	else:
		#use the full x if the model was trained with those stats
		p = model.predict(x)
	
	#saliency map calculation,
	#code from https://github.com/Harvard-IACS/2021-CS109B/blob/master/content/lectures/lecture17/notebook/lab8.ipynb
	with tf.GradientTape() as tape:
		modelOut, lastConvLayerOut = getMaps(x)
		
	grads = tape.gradient(modelOut, lastConvLayerOut)
	pooledGrads = tf.reduce_mean(grads, axis = (0, 1)) ##??
	reducedLast = tf.reduce_mean(tf.multiply(pooledGrads, lastConvLayerOut), 
								 axis=-1)
	reshapedLast = np.reshape(reducedLast.numpy(), -1)

	valsReshaped = (reshapedLast - reshapedLast.min())/(1e-7 + reshapedLast.max() - reshapedLast.min())
	
	saliencyMult = int(1000 / len(valsReshaped))
	
	saliency = np.repeat(valsReshaped, saliencyMult)
	
	#plot
	fig, (gsAx, pyAx, cAx) = plt.subplots(3)
		
	#plot chisquared p-values on log scale
	cAx.semilogy()
	cAx.plot(chi)
	
	#plot genotypes and saliency
	gsAx.stackplot([i for i in range(1000)],
				  x[0,:,1], #percent homo1
				  x[0,:,2], #percent het
				  x[0,:,3], #percent homo2
				  colors = ["#3FC4BE", "#ABC758", "#FECC46"]
				  )
	gsAx.plot(x[0,:,0], "#000000", alpha = 0.5)
	#saliency
	gsAx.plot(saliency, "w", alpha = 0.75)
	
	#plot predictions
	pyAx.plot(np.reshape(p, -1), "#674ea7")
	pyAx.plot(y, "#bf9000")
	
	#save
	plt.savefig(figName)
	
	plt.clf()

	print("wrote {}".format(figName))
	
	mult = 10
	
	#reshape everything so it can be written to a csv file
	#y and p need to be repeated because they are 100 long
	#while x is 1000 long
	xShaped = loaded
	yShaped = np.reshape(np.repeat(y, mult), (-1, 1))
	pShaped = np.reshape(np.repeat(np.reshape(p, -1), mult), (-1, 1))
	chiShaped = np.reshape(chi, (-1, 1))
	salShaped = np.reshape(saliency, (-1, 1))
		
	stats = np.concatenate([xShaped,yShaped,pShaped,chiShaped,salShaped], axis = 1)
	
	#write csv file
	with open(csvName, "w") as file:
		writer = csv.writer(file)
		for row in stats:
			writer.writerow(row)
