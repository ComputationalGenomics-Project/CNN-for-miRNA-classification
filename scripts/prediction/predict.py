########################################################################
#######       DO NOT MODIFY, DEFINITELY READ ALL OF THIS         #######
########################################################################

import numpy as np
import cnn_lenet
import pickle
import copy
import random
import seq2vec
import seq2mat
from sklearn.metrics import f1_score

def get_lenet():
  """Define LeNet

  Explanation of parameters:
  type: layer type, supports convolution, pooling, relu
  channel: input channel
  num: output channel
  k: convolution kernel width (== height)
  group: split input channel into several groups, not used in this assignment
  """

  layers = {}
  layers[1] = {}
  layers[1]['type'] = 'DATA'
  layers[1]['height'] = 128
  layers[1]['width'] = 19
  layers[1]['channel'] = 1
  layers[1]['batch_size'] = 64

  layers[2] = {}
  layers[2]['type'] = 'CONV'
  layers[2]['num'] = 20
  layers[2]['k'] = 5
  layers[2]['stride'] = 1
  layers[2]['pad'] = 0
  layers[2]['group'] = 1

  """layers[3] = {}
  layers[3]['type'] = 'POOLING'
  layers[3]['k'] = 2
  layers[3]['stride'] = 2
  layers[3]['pad'] = 0

  layers[4] = {}
  layers[4]['type'] = 'CONV'
  layers[4]['num'] = 50
  layers[4]['k'] = 5
  layers[4]['stride'] = 1
  layers[4]['pad'] = 0
  layers[4]['group'] = 1

  layers[5] = {}
  layers[5]['type'] = 'POOLING'
  layers[5]['k'] = 2
  layers[5]['stride'] = 2
  layers[5]['pad'] = 0"""

  layers[3] = {}
  layers[3]['type'] = 'IP'
  layers[3]['num'] = 500
  layers[3]['init_type'] = 'uniform'

  layers[4] = {}
  layers[4]['type'] = 'RELU'

  layers[5] = {}
  layers[5]['type'] = 'LOSS'
  layers[5]['num'] = 2
  return layers




def main():
  # define lenet
  layers = get_lenet()

  # load data
  # change the following value to true to load the entire dataset
  fullset = False
  """xtrain, ytrain, xval, yval, xtest, ytest = cnn_lenet.load_mnist(fullset)"""
  negPath = "/home/lui/CMU/Clinical-feature-learning/dataset/negative.txt"
  posPath = "/home/lui/CMU/Clinical-feature-learning/dataset/positive.txt"
  XTrain,yTrain = seq2mat.genTrain(posPath,negPath)
  print XTrain.shape, yTrain.shape
  threshold = XTrain.shape[1] / 10 * 9
  print "threshold: ", threshold
  XTrainTrue = XTrain[:,:threshold]
  Xtest = XTrain[:,threshold:]
  yTrainTrue = yTrain[:threshold]
  ytest = yTrain[threshold:]
  print "finish loading data"
  print "shape: xTrain, yTrain, xTest, yTest"
  print XTrainTrue.shape, yTrainTrue.shape, Xtest.shape, ytest.shape
  """xtrain = np.hstack([xtrain, xval])
  ytrain = np.hstack([ytrain, yval])"""
  m_train = XTrainTrue.shape[1]

  print "finish loading data"
  # cnn parameters
  batch_size = 32
  mu = 0.9
  epsilon = 0.01
  gamma = 0.0001
  power = 0.75
  weight_decay = 0.0005
  w_lr = 1
  b_lr = 2



  # initialize parameters
  pickle_path = 'lenet.mat'
  pickle_file = open(pickle_path, 'rb')
  params = pickle.load(pickle_file)
  param_winc = copy.deepcopy(params)

  for l_idx in range(1, len(layers)):
    param_winc[l_idx]['w'] = np.zeros(param_winc[l_idx]['w'].shape)
    param_winc[l_idx]['b'] = np.zeros(param_winc[l_idx]['b'].shape)

  layers[1]['batch_size'] = Xtest.shape[1]
  cp, param_grad, outcome = cnn_lenet.conv_net(params, layers, Xtest, ytest)
  
  vocab = {}
  vocab['tag'] = ["T", "F"]
  vocab['index'] = [0, 1]

  file_out = open("prediction.txt", 'w')
  file_out.write("True label" + '\t' + "Prediction\n")

  for i in range(len(outcome)):
    if outcome[i] in vocab['index']:
       file_out.write(vocab['tag'][outcome[i]])
       file_out.write('\t')
       file_out.write(vocab['tag'][ytest[i]])
       if i == len(outcome) - 1:
         break
       file_out.write('\n')
  pickle_file.close()
  file_out.close()
  fscore = f1_score(ytest, outcome)

  print '\ntest accuracy: %f\n' % (cp['percent'])
  print 'F1 score: %f\n' % (fscore)

if __name__ == '__main__':
  main()