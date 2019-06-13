import numpy as np
import operator
import os

def createDataset():
    group = np.array([[1.0,1.1], [1.0, 1.0], [0.0,0.0],[0.0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

samples, labels = createDataset()

def kNN(intX, samples, labels, k):
  dataSetSize = samples.shape[0]
  distMat = np.tile(intX, (dataSetSize, 1)) - samples
  distMat = distMat**2
  distances = distMat.sum(axis=1)
  distances = distances**0.5
  sortedDistance = distances.argsort()
  classCount = {}
  for i in range(0,k):
      label = labels[sortedDistance[i]]
      classCount[label] = classCount.get(label, 0)+1
  sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
  return sortedClassCount[0][0]

def img2vector(filename):
  vec = np.zeros((1,1024))
  with open(filename) as img_file:
    for i in range(32):
      data = img_file.readline()
      for j in range(32):
        vec[0,32*i+j] = int(data[j])
  return vec

def handewriteClassTest():
  trainingSampleFiles = os.listdir('trainingDigits')
  trainingSampleSize = len(trainingSampleFiles)
  trainingSamples = np.zeros((trainingSampleSize, 1024))
  trainingLabels = []
  for i in range(trainingSampleSize):
    fileName = trainingSampleFiles[i].split('.')[0]
    label = int(fileName.split('_')[0])
    trainingLabels.append(label)
    trainingSamples[i,:] = img2vector(os.path.join('trainingDigits', trainingSampleFiles[i]))
#    print("training %d is %d"%(i, label))
  
  testingSampleFiles = os.listdir('testDigits')
  testingSamplesSize = len(testingSampleFiles)
  errorCount = 0
  for i in range(testingSamplesSize):
    fileName = testingSampleFiles[i].split('.')[0]
    label = int(fileName.split('_')[0])
    testingSample = img2vector(os.path.join('testDigits', testingSampleFiles[i]))
    predict_label = kNN(testingSample, trainingSamples, trainingLabels, 3)
#    print("Predict %d is %d, actual %d"%(i, predict_label, label))
    if label != predict_label:
      errorCount = errorCount+1
  
  print("Error rate: %f"%(float(errorCount)/testingSamplesSize))

handewriteClassTest()