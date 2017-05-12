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

  # cnn parameters
  batch_size = 64
  mu = 0.9
  epsilon = 0.01
  gamma = 0.0001
  power = 0.75
  weight_decay = 0.0005
  w_lr = 1
  b_lr = 2

  test_interval = 500
  display_interval = 10
  snapshot = 100
  max_iter = 10000

  loadPara = bool(raw_input("load parameters, True or False: "))
  if loadPara:
    pickle_path = 'lenet.mat'
    pickle_file = open(pickle_path, 'rb')
    params = pickle.load(pickle_file)
    param_winc = copy.deepcopy(params)
  else:
  # initialize parameters
    params = cnn_lenet.init_convnet(layers)
    param_winc = copy.deepcopy(params)

  for l_idx in range(1, len(layers)):
    param_winc[l_idx]['w'] = np.zeros(param_winc[l_idx]['w'].shape)
    param_winc[l_idx]['b'] = np.zeros(param_winc[l_idx]['b'].shape)

  # learning iterations
  indices = range(m_train)
  random.shuffle(indices)
  for step in range(1500, max_iter):
    # get mini-batch and setup the cnn with the mini-batch
    start_idx = step * batch_size % m_train
    end_idx = (step+1) * batch_size % m_train
    if start_idx > end_idx:
      random.shuffle(indices)
      continue
    idx = indices[start_idx: end_idx]

    [cp, param_grad] = cnn_lenet.conv_net(params,
                                          layers,
                                          XTrainTrue[:, idx],
                                          yTrainTrue[idx])

    # we have different epsilons for w and b
    w_rate = cnn_lenet.get_lr(step, epsilon*w_lr, gamma, power)
    b_rate = cnn_lenet.get_lr(step, epsilon*b_lr, gamma, power)
    params, param_winc = cnn_lenet.sgd_momentum(w_rate,
                           b_rate,
                           mu,
                           weight_decay,
                           params,
                           param_winc,
                           param_grad)

    # display training loss
    if (step+1) % display_interval == 0:
      print 'cost = %f training_percent = %f' % (cp['cost'], cp['percent'])

    # display test accuracy
    if (step+1) % test_interval == 0:
      layers[1]['batch_size'] = Xtest.shape[1]
      cptest, _ = cnn_lenet.conv_net(params, layers, Xtest, ytest)
      layers[1]['batch_size'] = 64
      print '\ntest accuracy: %f\n' % (cptest['percent'])

    # save params peridocally to recover from any crashes
    if (step+1) % snapshot == 0:
      pickle_path = 'lenet.mat'
      pickle_file = open(pickle_path, 'wb')
      pickle.dump(params, pickle_file)
      pickle_file.close()


if __name__ == '__main__':
  main()
