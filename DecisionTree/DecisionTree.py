import math
import copy

def CreateDataSet():
  dataSet = [[1,1,'yes'],
           [1,1,'yes'],
           [1,0,'no'],
           [0,1,'no'],
           [0,1,'no']]
  labels = ['no surfacing', 'flippers']
  return dataSet, labels

def calShannonEnt(dataSet):
  total = len(dataSet)
  labelsDic = {}
  for data in dataSet:
    currentLabel = data[-1]
    if currentLabel in labelsDic:
      labelsDic[currentLabel] += 1;
    else:
      labelsDic[currentLabel] = 1;
  shannoEnt = 0.0
  for label in labelsDic:
    prob = float(labelsDic[label])/total
    shannoEnt -= prob*math.log(prob, 2)
  return shannoEnt 

def splitDataSet(dataSet, index, value):
  total = len(dataSet)
  retDataSet = []
  for data in dataSet:
    if data[index] == value:
      reducedFeatVec = data[:index]
      reducedFeatVec.extend(data[index+1:])
      retDataSet.append(reducedFeatVec)
#  print("splitDataSet " + str(retDataSet))
  return retDataSet

def chooseBestFeatureToSplit(dataSet):
  shannonEnt = calShannonEnt(dataSet)
  labelsCount = len(dataSet[0]) - 1
  bestInfoGain, bestFeature = 0.0, -1

  for i in range(labelsCount):
    featList = [sample[i] for sample in dataSet]
    featList = set(featList)

    labelEnt = 0.0
    for value in featList:
      vecs = splitDataSet(dataSet, i, value)
      prob = len(vecs)/float(len(dataSet))
      labelEnt += prob*calShannonEnt(vecs)
    
    if shannonEnt - labelEnt > bestInfoGain:
      bestInfoGain = shannonEnt - labelEnt
      bestFeature = i
  
  return bestFeature

def majorCnt(dataSet):
  countDic = {}
  for data in dataSet:
    if data[0] in countDic:
      countDic[data[0]] += 1
    else:
      countDic[data[0]] == 1
  return sorted(countDic.items(), key=lambda item:item[1])[-1]

def createTree(dataSet, labels):
#  print("Create tree "+str(dataSet))
  featList = [sample[-1] for sample in dataSet]
#  print("featList "+str(featList))
  if featList.count(featList[0]) == len(featList):
#    print("Return feat " + str(featList[0]))
    return featList[0]

  if len(dataSet[0]) == 1:
    return majorCnt(featList)
  
  bestFeat = chooseBestFeatureToSplit(dataSet)
#  print("Best feat " + str(bestFeat))
  bestLabel = labels[bestFeat]
  newTree = {bestLabel:{}}

  del(labels[bestFeat])
  featValues = [sample[bestFeat] for sample in dataSet]
  featValues = set(featValues)

  for value in featValues:
    sublabels = labels[:]
    newTree[bestLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), sublabels)
  
  return newTree

def classfy(inputTree, featLabels, testVec):
  rootLabel = inputTree.keys()[0]
  childTree = inputTree[rootLabel]
  labelIndex = featLabels.index(rootLabel)

  key = testVec[labelIndex]
  valueOfFeat = childTree[key]
  print '+++', rootLabel, 'xxx', childTree, '---', key, '>>>', valueOfFeat
  if isinstance(valueOfFeat, dict):
    classLabel = classfy(valueOfFeat, featLabels, testVec)
  else:
    classLabel = valueOfFeat
  return classLabel


trainingSet, labels = CreateDataSet()
tree = createTree(trainingSet, copy.deepcopy(labels))
testSet = [0,0]
print(classfy(tree, labels, testSet))
print(classfy(tree, labels, [1,0]))
print(classfy(tree, labels, [1,1]))
print(classfy(tree, labels, [0,1]))