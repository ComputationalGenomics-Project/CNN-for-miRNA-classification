import numpy as np
import seq2vec

"""get a np matrix from input data"""
def openFile(fname):
    inputFile = [i.strip() for i in open(fname).readlines()]
    inputFile = np.array(inputFile)
    return inputFile

"""convert sequence to matrix with 64 cols"""
def seq2mat(sequence):
    hotMat = seq2vec.seq2vec(sequence).reshape(len(sequence)-2,64)
    return hotMat

"""convert sequence to matrix with 128 rows and 19 cols"""
def genMat(sequence):
    top = seq2mat(sequence[:len(sequence)-1]).T
    bottom = seq2mat(sequence[1:]).T
    combined = np.vstack((top,bottom))
    if combined.shape[1] > 19:
        combined = combined[:,:19]
    if combined.shape[1] < 19:
        addNum = 19 - combined.shape[1]
        added = np.zeros((128,addNum))
        combined = np.hstack((combined,added))
    return combined

"""load all the data in file into a 2-D matrix, row number is len(data) and col number is 128*19 = 2432,
each row is a sequence"""
def file2Mat(fname):
    XTrain = []
    allData = openFile(fname)
    for i in range(len(allData)):
        data = genMat(allData[i]).reshape(-1,1)
        XTrain.append(data)
    XTrain = np.array(XTrain).reshape(len(XTrain),128 * 19)
    return XTrain

"""get both XTrain and yTrain"""
def genTrain(pos,neg):
    yTrain = []
    XPositive = file2Mat(pos)
    XNegative = file2Mat(neg)
    for i in range(XPositive.shape[0]):
        yTrain.append(1)
    for i in range(XNegative.shape[0]):
        yTrain.append(0)
    XTrain = np.vstack((XPositive,XNegative))
    yTrain = np.array(yTrain)
    return XTrain,yTrain

"""make XTrain be a 3-D matrix, first dimension is 128 * 19, representing a sequence, second dimension is 64,
represneting batch size and the last dimension is len(allData) / 64, representing how many batches in total"""
def addBatch(pos,neg,batchSize = 64):
    XTrain,yTrain = genTrain(pos,neg)
    batchNum = XTrain.shape[0] / batchSize
    allItem = batchNum * batchSize
    XTrain = XTrain[:allItem,:]
    yTrain = yTrain[:allItem]
    XTrain = XTrain.reshape(128*19,batchSize,batchNum)
    yTrain = yTrain.reshape(batchSize,batchNum)
    return XTrain,yTrain

#XTrain,yTrain = addBatch('positive.txt','negative.txt')