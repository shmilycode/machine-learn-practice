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
    print(trainMatrix[i])
    if trainCategory[i] == 1:
      p1Num += trainMatrix[i]
      p1Denom += sum(trainMatrix[i])
    else:
      p0Num += trainMatrix[i]
      p0Denom += sum(trainMatrix[i])
  print(p0Num, p0Denom)
  print(p1Num, p1Denom)
  p0Vec = math.log(p0Num/p0Denom)
  p1Vec = math.log(p1Num/p1Denom)
  return p0Vec, p1Vec, pAbusive

trainingList, trainCategory = loadDataSet()
vocabList = createVocabList(trainingList)
trainingSamples = []
for vec in trainingList:
  trainingSamples.append(setOfWords2Vec(vocabList, vec))
p0Vec, p1Vec, pAbusive = trainNB0(trainingSamples, trainCategory)
print(p0Vec)
print(p1Vec)
print(pAbusive)