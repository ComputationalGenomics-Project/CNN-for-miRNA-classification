import numpy as np
import seq2vec

"""get a np matrix from input data"""
def openFile(fname):
    inputFile = [i.strip() for i in open(fname).readlines()]
    inputFile = np.array(inputFile)
    return inputFile

"""convert sequence to matrix with 64 coloums"""
def seq2mat(sequence):
    hotMat = seq2vec.seq2vec(sequence).reshape(len(sequence)-2,64)
    return hotMat

"""convert sequence to matrix with 128 rows"""
def genMat(sequence):
    top = seq2mat(sequence[:len(sequence)-1]).T
    bottom = seq2mat(sequence[1:]).T
    combined = np.vstack((top,bottom))
    return combined

def trainCNN(sequence):

    return

"""a = openFile('out.txt')
print genMat(a[0])[:,0]"""