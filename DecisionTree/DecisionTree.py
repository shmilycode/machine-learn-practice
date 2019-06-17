import math
import copy
import matplotlib.pyplot as plt

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

def majorCnt(classList):
  countDic = {}
  for data in classList:
    if data not in countDic:
      countDic[data] = 0
    countDic[data] += 1
  return sorted(countDic.items(), key=lambda item:item[1])[-1][0]

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
  print('+++', rootLabel, 'xxx', childTree, '---', key, '>>>', valueOfFeat)
  if isinstance(valueOfFeat, dict):
    classLabel = classfy(valueOfFeat, featLabels, testVec)
  else:
    classLabel = valueOfFeat
  return classLabel

def retrieveTree(i):
  listOfTrees = [{'no surfacing': {0: 'no', 1:{'flippers': \
    {0: 'no', 1:'yes'}}}},
    {'no surfacing':{0:'no',1:{'flippers':\
      {0:{'head':{0:'no',1:'yes'}}, 1:'no'}}}}]
  return listOfTrees[i]

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeText, centerPt, parentPt, nodeType):
  createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords='axes fraction',
  xytext=centerPt, textcoords='axes fraction',
  va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

#def createPlot():
#  fig = plt.figure(1, facecolor='white')
#  fig.clf()
#  createPlot.ax1 = plt.subplot(111, frameon=False)
#  plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#  plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#  plt.show()

def getNumLeafs(myTree):
  numLeafs = 0
  firstStr = list(myTree.keys())[0]
  secondDict = myTree[firstStr]
  for key in secondDict.keys():
    if isinstance(secondDict[key], dict):
      numLeafs += getNumLeafs(secondDict[key])
    else:
      numLeafs += 1
  return numLeafs

def getTreeDepth(myTree):
  maxDepth = 0
  firstStr = list(myTree.keys())[0]
  secondDict = myTree[firstStr]
  for key in secondDict.keys():
    if isinstance(secondDict[key], dict):
      thisDepth = 1 + getTreeDepth(secondDict[key])
    else:
      thisDepth = 1
    if thisDepth > maxDepth : 
      maxDepth = thisDepth
  return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
  xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
  yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
  createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
  numLeafs = getNumLeafs(myTree)
  depth = getTreeDepth(myTree)
  firstStr = list(myTree.keys())[0]
  cntrPt = (plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW, \
    plotTree.yOff)
  plotMidText(cntrPt, parentPt, nodeTxt)
  plotNode(firstStr, cntrPt, parentPt, decisionNode)
  secondDict = myTree[firstStr]
  plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
  for key in secondDict.keys():
    if isinstance(secondDict[key], dict):
      plotTree(secondDict[key], cntrPt, str(key))
    else:
      plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
      plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),
        cntrPt, leafNode)
      plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
  plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
  fig = plt.figure(1, facecolor = 'white')
  fig.clf()
  axprops = dict(xticks=[], yticks=[])
  createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
  plotTree.totalW = float(getNumLeafs(inTree))
  plotTree.totalD = float(getTreeDepth(inTree))
  plotTree.xOff = -0.5/plotTree.totalW
  plotTree.yOff = 1.0
  print("totalw(%d), totalD(%d), xOff(%d), yOff(%d)"%(plotTree.totalW, plotTree.totalD, plotTree.xOff, plotTree.yOff))
  plotTree(inTree, (0.5,1.0), '')
  plt.show()

#1. 
#myTree = retrieveTree(1)
#createPlot(myTree)

#2.
#createPlot()

#3. 
#trainingSet, labels = CreateDataSet()
#tree = createTree(trainingSet, copy.deepcopy(labels))
#testSet = [0,0]
#print(classfy(tree, labels, testSet))
#print(classfy(tree, labels, [1,0]))
#print(classfy(tree, labels, [1,1]))
#print(classfy(tree, labels, [0,1]))

#4.
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
print(lensesTree)
createPlot(lensesTree)