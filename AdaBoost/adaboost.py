import numpy as np
from numpy import *
import matplotlib.pyplot as plt
def loadSimpData():
  datMat = np.matrix([[1.,2.1],
  [2.,1.1],
  [1.3,1.],
  [1.,1.],
  [2.,1.]])
  classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
  return datMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
  retArray = np.ones((np.shape(dataMatrix)[0], 1))
  if threshIneq == 'lt':
    retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
  else:
    retArray[dataMatrix[:,dimen] > threshVal] = -1.0
  return retArray

def buildStump(dataArr, classLabels, D):
  dataMatrix = np.mat(dataArr)
  classMatrix = np.mat(classLabels).T
  m,n = np.shape(dataMatrix)
  numSteps = 10.0; bestStump = {}; bestClassEst = np.mat(np.zeros((m,1)))
  minError = np.inf
  for i in range(n):
    minVal = dataMatrix[:,i].min()
    maxVal = dataMatrix[:,i].max()
    gap = (maxVal - minVal)/numSteps
    for threshIndex in range(-1, int(numSteps) + 1):
      for inequal in ['lt', 'gt']:
        threshVal = minVal + float(threshIndex)*gap
        predictVals = stumpClassify(dataMatrix, i, threshVal, inequal)
        errArr = np.mat(np.ones((m,1)))
        errArr[predictVals == classMatrix] = 0
        weightedError  = D.T*errArr
        if weightedError < minError:
          minError = weightedError
          bestClassEst = predictVals.copy()
          bestStump['dim'] = i
          bestStump['thresh'] = threshVal
          bestStump['ineq'] = inequal
  return bestStump, minError, bestClassEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
  weakClassArr = []
  m = np.shape(dataArr)[0]
  D = np.mat(np.ones((m,1))/m)
  aggClassEst = np.mat(np.zeros((m, 1)))
  for i in range(numIt):
    bestStump, error, classEst = buildStump(dataArr, classLabels, D)
    alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))
#    print("D:",D.T)
#    print("alpha:",alpha)
#    print("classEst: ",classEst.T)
    bestStump['alpha'] = alpha
    weakClassArr.append(bestStump)
    expon = np.multiply(-1*alpha*np.mat(classLabels).T, classEst)
#    print("expon: ",expon)
    D = np.multiply(D, np.exp(expon))
    D = D/D.sum()
    aggClassEst += alpha * classEst
    aggErrors = np.multiply(np.sign(aggClassEst)!=
                np.mat(classLabels).T, np.ones((m,1)))
    errorRate = aggErrors.sum()/m
    print("total error: ", errorRate)
    if errorRate == 0.0:
      break
  return weakClassArr,aggClassEst

def adaClassify(dataToClass, classifierArr):
  dataMatrix = np.mat(dataToClass)
  m = np.shape(dataMatrix)[0]
  aggClassEst = np.mat(np.zeros((m,1)))
  for i in range(len(classifierArr)):
    classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
      classifierArr[i]['thresh'], classifierArr[i]['ineq'])
    aggClassEst += classifierArr[i]['alpha'] * classEst
#    print(aggClassEst)
  return np.sign(aggClassEst)
#datMat, classLabels = loadSimpData()
#classifierArray = adaBoostTrainDS(datMat, classLabels, 9)
#print(adaClassify([[5,5],[0,0]], classifierArray))

def plotROC(predStrengths, classLabels):
  cur = (1.0, 1.0)
  ySum = 0.0
  numPosClass = np.sum(array(classLabels) == 1.0)
  yStep = 1/float(numPosClass)
  xStep = 1/float(len(classLabels) - numPosClass)
  sortedIndicies = predStrengths.argsort()
  fig = plt.figure()
  fig.clf()
  ax = plt.subplot(111)
  for index in sortedIndicies.tolist()[0]:
    if classLabels[index] == 1.0:
      delX = 0; delY = yStep;
    else:
      delX = xStep; delY = 0;
      ySum += cur[1]
    ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
    cur = (cur[0] - delX, cur[1] - delY)
  ax.plot([0,1],[0,1], 'b--')
  plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
  plt.title('ROC')
  ax.axis([0,1,0,1])
  plt.show()
  print("AUC=", ySum*xStep)

def loadDataSet(filename):
  numFeat = len(open(filename).readline().split('\t'))
  dataMat = []; labelMat = []
  fr = open(filename)
  for line in fr.readlines():
    lineArr = []
    curLine = line.strip().split('\t')
    for i in range(numFeat - 1):
      lineArr.append(float(curLine[i]))
    dataMat.append(lineArr)
    labelMat.append(float(curLine[-1]))
  return dataMat, labelMat


datArr, labelArr = loadDataSet('horseColicTraining2.txt')
classifierArray, aggClassEst = adaBoostTrainDS(datArr, labelArr, 10)
plotROC(aggClassEst.T, labelArr)