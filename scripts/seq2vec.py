# convert sequences to one-hot word vectors. 
#
# @author Luyi Ma
#
import math
import numpy as np

def getATCG(idx):
	atcg = ['A', 'T', 'G', 'C']
	if idx < 0 or idx > 3:
		return 'X'
	return atcg[idx]

def pull_permute(cur, all, array, seq):
	if cur == all:
		array.append(seq)
		return
	for i in range(4):
		seq = seq + getATCG(i)
		pull_permute(cur + 1, all, array, seq)
		seq = seq[: -1]


def getLookupTable(length = 3):
	if (length < 3):
		length = 3
	num = int(math.pow(4, length))
	array = []
	seq = ''
	cur = 0
	pull_permute(cur, length, array, seq)
	lookup = {}
	lookup['kmer'] = array
	table = []
	for i in range(len(array)):
		temp = [0 for i in range(len(array))]
		temp[i] = 1
		table.append(temp)
	lookup['table'] = table
	return lookup 



def seq2vec(seq, stride = 1):
	k = 3
	lookup = getLookupTable(k)
	num = len(lookup['kmer'])
	vec = []

	if seq != None:
		len_out = (len(seq) - k) / stride + 1
		assert len_out > 0, 'invalid seq length or stride size'
		for i in range(len_out):
			temp_Seq = seq[i: i+k]
			if temp_Seq in lookup['kmer']:
				vec = np.hstack(vec, lookup['table'][lookup['kmer'].index(temp_Seq)])
			else:
				vec = np.hstack(vec, [0 for i in range(num)])
	return vec

def load_data():
	path = raw_input("data: ")
	file = open(path, 'r')
	sequence = []
	label = []
	while (True):
		line = file.readline()
		if line == "":
			break
		line = line.split("\t")
		sequence.append(line[0])
		label.append(int(line[1][:-1]))
	seq_img = np.array([seq2vec(i, 1) for i in sequence])
	yLabel = np.array(label)

	# shuffle
	shuffle_indices = range(seq_img.shape[0])
	np.random.shuffle(shuffle_indices)
	seq_img.take(shuffle_indices, axis = 0, out = seq_img)
	yLabel.take(shuffle_indices, axis = 0, out = yLabel)

	# partition
	trainNum = int(math.floor(seq_img.shape[0] / 5 * 4))
	xTrain = seq_img[: trainNum]
	yTrain = yLabel[: trainNum]
	xTest = seq_img[trainNum: ]
	yTest = yLabel[trainNum: ]

	# transpose
	xTrain = xTrain.T
	xTest = xTest.T

	return (xTrain, yTrain, xTest, yTest)





