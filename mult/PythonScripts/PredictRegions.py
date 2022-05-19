#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 17:39:04 2022

@author: liathomson

CNN to predict where things are on chromosomes
"""
#import statements
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization as layerNormalization
from MultDataGen import ChrGen


#if testing
if(False):
	argv = ["PredictRegions.py", "outMulti1.npz", "outMulti1.npz", "outMulti1.npz",
		 "testPredict", "0", "0", "11", "5", "16", "16", "100", "0"]
	
	print("TESTING SCRIPT", flush = True)
	
	genBatchSize = 6
	normBatchSize = 24
#if genuinely running
else:
	from sys import argv
	
	genBatchSize = 144
	normBatchSize = 7200

#gather all command-line arguments
trainGlob = argv[1]	 #glob string to get all training NPZs
valGlob = argv[2]	   #glob string to get all validation NPZs
neutralGlob = argv[3]   #used to configure normalization
modelPath = argv[4]	 #file to save the model to
version = int(argv[5]) #version of the model
						#assumes model actual path is modelPath_v{version}
load = bool(int(argv[6])) #whether to load a model and continue its
							#training

kernel1 = int(argv[7])   #somewhere in the range of 3 and 17, maybe
kernel2 = int(argv[8])  #yup
filters1 = int(argv[9])	#num filters per convolution
filters2 = int(argv[10])
epochs = int(argv[11])   #num epochs to train

excludeStats = bool(int(argv[12]))

print("called with arguments\n"
	  "\ttrainGlob: {}\n"
	  "\tvalGlob: {}\n"
	  "\tneutralGlob: {}\n"
	  "\tmodelPath: {}\n"
	  "\tversion: {}\n"
	  "\tload: {}\n"
	  "\tkernel1: {}\n"
	  "\tkernel2: {}\n"
	  "\tfilters1: {}\n"
	  "\tfilters2: {}\n"
	  "\tepochs: {}\n"
	  "\texcludeStats: {}\n".format(
		trainGlob,
		valGlob,
		neutralGlob,
		modelPath,
		version,
		load,
		kernel1,
		kernel2,
		filters1,
		filters2,
		epochs,
		excludeStats
	),
	flush = True)


from glob import glob
print("train: {}".format(glob(trainGlob)))

print("validation: {}".format(glob(valGlob)))


print("neutral: {}".format(glob(neutralGlob)))



#data generators
trainGen = ChrGen(trainGlob, 
				  batchSize = genBatchSize, 
				  excludeStats = excludeStats)
valGen = ChrGen(valGlob, 
				batchSize = genBatchSize, 
				excludeStats = excludeStats)

print("training and validation generators loaded")

#load or create model
if(load):
	model = tf.keras.models.load_model("{}_v{}".format(modelPath, version))
else:
	
	#preprocessing
	#normalization layer
		
	if(excludeStats):
		nInitialStats = 4
	else:
		nInitialStats = 7
	
	
	neutralGen = ChrGen(neutralGlob, 
						 batchSize = normBatchSize, 
						 excludeStats = excludeStats)

	print(neutralGen.npzFiles)
	print(len(neutralGen))

	normalizeLayer = layerNormalization(name = "normalization",
										 input_shape = (1000, nInitialStats))
	normalizeLayer.adapt(neutralGen[0][0])
	
	print("normalization layer configured")
	
	
	#create model
	model = tf.keras.models.Sequential([
		normalizeLayer,
		
		layers.Conv1D(filters1, kernel1, 
						padding = "same", activation = "relu",
						name = "conv1d_a1"),
		layers.Conv1D(filters1, kernel1, 
						padding = "same", activation = "relu",
						name = "conv1d_a2"),
		layers.AveragePooling1D(5,
						  padding = "same",
						  name = "avgPool1d_a"),
		layers.Conv1D(filters1, kernel1, 
						padding = "same", activation = "relu",
						name = "conv1d_a3"),
		layers.Conv1D(filters1, kernel1, 
						padding = "same", activation = "relu",
						name = "conv1d_a4"),
		layers.MaxPool1D(2, 
					   padding = "same",
					   name = "maxPool1d_a"),
		
		layers.Conv1D(filters2, kernel2, 
						padding = "same", activation = "relu",
						name = "conv1d_b1"),
		layers.Conv1D(filters2, kernel2, 
						padding = "same", activation = "relu",
						name = "conv1d_b2"),
		layers.Conv1D(1, kernel2, 
						padding = "same",
						name = "conv1d_b3"),
		layers.Flatten(name = "flatten_b"),
		layers.ReLU(name = "relu_b", max_value = 1.0, threshold = 0.0)
		])
		
	
	#compile
	model.compile(optimizer = tf.keras.optimizers.Adam(),
				loss = tf.keras.losses.MeanSquaredError(),
				metrics = [tf.keras.metrics.MeanAbsoluteError()]
				)

	
#print out model summary
print("\n\n---------- Model ----------\n")
model.summary()


#callback (to stop early)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor = "val_loss",
											  min_delta = 1e-4,
											  patience = 10,
											  restore_best_weights = True)]

print("\n\n---------- Training ----------\n")

#train
histObj = model.fit(trainGen,
					epochs = epochs,
					callbacks = callbacks,
					validation_data = valGen
					)

print("\n\n---------- Evaluation ----------\n")

#final evaluation
evalLoss, evalMet = model.evaluate(valGen)
print("mean squared error: {}, mean absolute error: {}".format(evalLoss, 
															   evalMet))

print("\n\n---------- Finishing ----------\n")

saveModelPath = "{}_v{}".format(modelPath, version + 1)

print("Saving model to {}...".format(saveModelPath))

#save
model.save(saveModelPath)