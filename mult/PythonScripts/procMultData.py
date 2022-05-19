#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#takes a csv file
#reads all the simulation CSVs from it
#saves their data to a .npz

from sys import argv

import numpy as np
from scipy.stats import multinomial, chisquare #, power_divergence

import csv

class AncestryMatrix:
	#calculates and stores a matrix of the ancestry of all F2 individuals
	#no previous generations
	
	def __init__(self, ancCSV):
		self.numChr = 6
		
		#read and reformat ancestry csv file
		ancRows = []
		with open(ancCSV, "r") as file:
			reader = csv.reader(file)
			
			for row in reader:
				ancRows.append(row)
			
			chrLength = len(row)
			
		ancArray = np.array(ancRows).astype(int)
		
		#add data to self.
		self.chrLength = chrLength
		self.nInds = np.sum(ancArray[:, 0])
		
		self.sumHomo1 = ancArray[0]
		self.sumHet = ancArray[1]
		self.sumHomo2 = ancArray[2]
		
		self.meanHomo1 = self.sumHomo1 / self.nInds
		self.meanHet = self.sumHet / self.nInds
		self.meanHomo2 = self.sumHomo2 / self.nInds
		
		#note: meanAnc is specifically ancestry for p2
		self.meanAnc = (self.sumHomo2 + (self.sumHet * 0.5)) / self.nInds
				
		#stats
		#self.multiPun = np.full(chrLength, -1.0)
		self.multiPQ = np.full(chrLength, -1.0)
		#self.chiPun = np.full(chrLength, -1.0)
		#self.chiPunP = np.full(chrLength, -1.0)
		self.chiPQ = np.full(chrLength, -1.0)
		self.chiPQP = np.full(chrLength, -1.0)
				
		#self.y = np.zeros(int(self.chrLength / 10))
		self.y = np.zeros(self.chrLength)
		
		self.statsCalc = False
				
	def calcAllStats(self, recalculate = False):
		#this function is time-consuming, so check to make sure
		#it needs to be run
		if(self.statsCalc and not recalculate):
			raise Exception("Stats already calculated.")
		
		#reformat the genotypes to work better with the functions
		#so each row is the genotypes for a particular locus
		actualSum = np.transpose([self.sumHomo1, self.sumHet, self.sumHomo2])

		#punExpect = np.array([0.25, 0.5, 0.25])
		
		#get expected frequencies
		#based off Hardy-Weinberg equilibrium
		#where p = p2 ancestry and q = p1 ancestry
		p = self.meanAnc
		q = 1 - p
		
		#make an array of expected frequencies of each genotype
		calcPqExpect = np.frompyfunc(lambda p, q: np.array([p ** 2, 2 * p * q, q ** 2]),
									 2, 1)
		pqExpect = calcPqExpect(p, q)
		
		#make an array of multinomial objects, based off the expected frequencies
		getMultinom = np.frompyfunc(lambda pqExpect : 
									  multinomial(self.nInds, pqExpect),
									1, 1)
		multinomials = getMultinom(pqExpect)
			  
		#go through every position on the chromosome to calculate
		#multinomial likelihood and chisquared likelihood of the
		#actual frequencies of genotypes
		for i in range(self.chrLength):
			self.multiPQ[i] = multinomials[i].pmf(actualSum[i])
			self.chiPQ[i], self.chiPQP[i] = chisquare(actualSum[i],
													  pqExpect[i] * self.nInds)
				
		#mark stats as calculated
		self.statsCalc = True
	
	def getLongStats(self):
		#reformat compressed stats into long stats
		if(self.statsCalc):
			stacked = np.stack([self.meanAnc, 
								self.meanHomo1, self.meanHet, self.meanHomo2,
								self.multiPQ, self.chiPQ, self.chiPQP], axis = 1)
			#split into 
			split = np.split(stacked, self.numChr, axis = 0)
			return split
							
		else:
			raise Exception("Not all stats have been calculated.")
	
	def calcYLoc(self, f, p):		
		#select from f only the fitness effect
		fit = np.min(f, axis = 0)
		
		#bin pos
		#pos = np.round(p / 10).astype(int)
		pos = p
				
		for i in range(p.shape[0]):
			for j in range(2):
				self.y[pos[i, j]] = 1.0 - fit[i]
	
	def getY(self):
		split = np.split(self.y, self.numChr, axis = 0)
		return split


inputCSV = "testMultDMI1.csv"
outName = "outMulti1"
inputCSV = argv[1]  #name of the CSV file that contains all the relevant info.
outName = argv[2]   #name of file to write the .npz to

print("analyzing {} and outputting to {}".format(inputCSV, outName), 
	  flush = True)

sArr = []
fArr = []
pArr = []
yArr = []

with open(inputCSV, "r") as cfile:
	reader = csv.reader(cfile)
	#each row is made of:
	#tree file name, seed, K, N, m1Loc, m2Loc, f1, f2, f3, f4, f5, f6, f7, f8, f9,
	#num1, num2, num3, num4, num5, num6, num7, num8, num9
	#(K is number F0 individuals and max. F1 individuals, N is number F2 individuals)
	for row in reader:
		#read the row

		#seed = row[0], ignore
		simName = row[1]
		print("analyzing {}...".format(simName), flush = True)
		#ancIndName = row[2]
		#saveIndData = row[3]
		fitName = row[4]
		posName = row[5]

		#get fitness information
		f = []
		with open(fitName, "r") as fitFile:
			reader = csv.reader(fitFile)
			for row in reader:
				f.append(row)
		f = np.array(f).astype(float)
				
		#get position information
		p = []
		with open(posName, "r") as posFile:
			reader = csv.reader(posFile)
			for row in reader:
				p.append(row)	
		p = np.array(p).astype(int)

		#read the tree, get stats
		am = AncestryMatrix(simName)
		am.calcAllStats()
		am.calcYLoc(f, p)
		s = am.getLongStats()
		y = am.getY()

		#append to arrays
		sArr.extend(s)
		yArr.extend(y)
		
		#fArr.append(f)
		#pArr.append(p)

#save all input and outputs to a compressed .npz file
np.savez(outName,
		  	s = np.array(sArr),
			#f = np.array(fArr),
			#p = np.array(pArr),
			y = np.array(yArr))
