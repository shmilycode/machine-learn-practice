import numpy as np
import math
def loadDataSet():
  postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
  classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
  return postingList,classVec

def createVocabList(dataSet):
  vocabSet = set([])
  for vec in dataSet:
    vocabSet = vocabSet | set(vec)
  return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
  returnVec = [0]*len(vocabList)
  for word in inputSet:
    if word in vocabList:
      returnVec[vocabList.index(word)] = 1
#    else:
#      print("word %s not in vocablist"%word)
  return returnVec

def trainNB0(trainMatrix, trainCategory):
  numMatrix = len(trainMatrix)
  numWords = len(trainMatrix[0])
  pAbusive = sum(trainCategory)/numMatrix

  p0Num = np.ones(numWords)
  p1Num = np.ones(numWords)
  p0Denom = 2.0
  p1Denom = 2.0
  for i in range(numMatrix):
    if trainCategory[i] == 1:
      print("1: ", trainMatrix[i])
      p1Num += trainMatrix[i]
      p1Denom += sum(trainMatrix[i])
    else:
      print("0: ", trainMatrix[i])
      p0Num += trainMatrix[i]
      p0Denom += sum(trainMatrix[i])
  print(p0Num, p0Denom)
  print(p1Num, p1Denom)
  p0Vec = np.log(p0Num/p0Denom)
  p1Vec = np.log(p1Num/p1Denom)
  return p0Vec, p1Vec, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
  p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)
  p0 = sum(vec2Classify*p0Vec) + np.log(pClass1)
  if p1 > p0:
    return 1
  else:
    return 0

def bagOfWords2VecMn(vocabList, inputSet):
  returnVec = [0]*len(vocabList)
  for word in inputSet:
    if word in inputSet:
      returnVec[vocabList.index(word)] += 1

  return returnVec

def testingNB():
  datas, labels = loadDataSet()
  myVocabList = createVocabList(datas)
  trainMat = []
  for data in datas:
    trainMat.append(bagOfWords2VecMn(myVocabList, data))
  p0, p1, pAb = trainNB0(np.array(trainMat), np.array(labels))
  testEntry = ['love', 'my', 'dalmation']
  thisDoc = np.array(bagOfWords2VecMn(myVocabList,testEntry))
  print("classify as: ", classifyNB(thisDoc, p0, p1, pAb))
  testEntry = ['stupid', 'garbage']
  thisDoc = np.array(bagOfWords2VecMn(myVocabList,testEntry))
  print("classify as: ", classifyNB(thisDoc, p0, p1, pAb))

testingNB()