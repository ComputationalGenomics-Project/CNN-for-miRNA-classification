import numpy as np
import matplotlib.pyplot as plt
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
    yTrain = np.matrix(np.array(yTrain))
    yTrain = yTrain.T
    allTrain = np.hstack((XTrain,yTrain))
    np.random.shuffle(allTrain)
    XTrain = allTrain[:,:allTrain.shape[1]-1]
    yTrain = allTrain[:,allTrain.shape[1]-1:]
    return XTrain.T,yTrain

"""make XTrain be a 3-D matrix, first dimension is 128 * 19, representing a sequence, second dimension is 64,
represneting batch size and the last dimension is len(allData) / 64, representing how many batches in total
def addBatch(pos,neg,batchSize = 64):
    XTrain,yTrain = genTrain(pos,neg)
    batchNum = XTrain.shape[0] / batchSize
    allItem = batchNum * batchSize
    XTrain = XTrain[:allItem,:]
    yTrain = yTrain[:allItem]
    XTrain = XTrain.reshape(128*19,batchSize,batchNum)
    yTrain = yTrain.reshape(batchSize,batchNum)
    return XTrain,yTrain"""

#XTrain,yTrain = genTrain('positive.txt','negative.txt')
seq1 = 'TGAGGTAGTAGGTTGTATAGTT'
seq2 = 'TGGAATGTAAAGAAGTATGTA'
seq3 = 'TGCTGGTTTCTTCCACAGTGGTA'
seq4 = 'TCTTACTCATCCTATTCTTTAA'
seq5 = 'CATTAACCTTCATATTTCCTT'
seq6 = 'CATTTTAAATATTACCTCTATT'

mat1 = genMat(seq1)

mat1 = mat1.T
im = plt.imshow(mat1, cmap = 'bwr', origin = 'lower')
plt.colorbar(im, orientation='vertical')
plt.show()


mat2 = genMat(seq2)

mat2 = mat2.T
im = plt.imshow(mat2, cmap = 'bwr', origin = 'lower')
plt.colorbar(im, orientation='vertical')
plt.show()

mat3 = genMat(seq3)

mat3 = mat3.T
im = plt.imshow(mat3, cmap = 'bwr', origin = 'lower')
plt.colorbar(im, orientation='vertical')
plt.show()

mat4 = genMat(seq4)
print mat1.shape
mat4 = mat4.T
im = plt.imshow(mat4, cmap = 'bwr', origin = 'lower')
plt.colorbar(im, orientation='vertical')
plt.show()

mat5 = genMat(seq5)
print mat1.shape
mat5 = mat5.T
im = plt.imshow(mat5, cmap = 'bwr', origin = 'lower')
plt.colorbar(im, orientation='vertical')
plt.show()

mat6 = genMat(seq6)
print mat1.shape
mat6 = mat6.T
im = plt.imshow(mat6, cmap = 'bwr', origin = 'lower')
plt.colorbar(im, orientation='vertical')
plt.show()

#file = open("vectorize.txt", 'w')
