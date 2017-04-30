import random as rd
from datetime import datetime
import math
import numpy as np


def permute(m = 8):
	file = open("permutation.txt", 'w')
	pathref = raw_input("Input the miRNA file: ")
	fileref = open(pathref, 'r')
	mir = []
	while (True):
		line = fileref.readline()
		if (line != ''):
			break
		mir.append(line[:-1])
	print "finish loading miRNA sequences"
	if (m < 8):
		m = 8
	roundNum = int(math.pow(4, m))
	count = 0
	print "roundNum: ", roundNum
	for i in xrange(roundNum):
		if (i % 1000 == 0):
			print "round: ", i
		length = getLen()
		tempStr = ''
		for i in range(length):
			tempStr += ATCG()
		if (tempStr in mir):
			continue
		boolean = False
		for j in mir:
			if (j in tempStr):
				boolean = True
				break
		if (boolean):
			continue
		file.write(tempStr + '\n')
		count += 1
	file.close()
	fileref.close()
	print "Number of Permutations: ", count


def ATCG():
	#model = [0.24047454, 0.28349688, 0.21761676, 0.25841182]
	rd.seed(datetime.now())
	temp = rd.random()
	if (temp <= 0.25):
		return 'A'
	elif (temp <= 0.5):
		return 'T'
	elif (temp <= 0.75):
		return 'C'
	else:
		return 'T'


def getLen():
	stats = [108, 366, 818, 2683, 10518, 12580, 5349, 2482, 666, 141, 54]
	model = np.array(stats) * 1.0 / np.sum(stats)
	rd.seed(datetime.now())
	temp = rd.random()
	x = [i for i in range(17, 28)]
	count = 0
	for i in range(len(x)):
		count += model[i]
		if (temp <= count):
			return x[i]

permute(8)