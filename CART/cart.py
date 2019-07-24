import numpy as np

def regLeaf(dataSet):
  return np.mean(dataSet[:,-1])

def regErr(dataSet):
  return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

def binSplitDataSet(dataSet, feature, value):
  mat0 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
  mat1 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
  return mat0, mat1

def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops=(1,4)):
  tolS = ops[0]
  tolN = ops[1]

  if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
    return None, leafType(dataSet)
  
  m,n = np.shape(dataSet)
  S = errType(dataSet)
  bestS, bestIndex, bestValue = np.inf, 0, 0
  for featIndex in range(n-1):
    for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):
      mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
      if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        continue;
      newS = errType(mat0) + errType(mat1)
      if newS < bestS:
        bestIndex = featIndex
        bestValue = splitVal
        bestS = newS
  
  if S - bestS < tolS:
    return None, leafType(dataSet)
  
  mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
  if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
    return None, leafType(dataSet)
  return bestIndex, bestValue

def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
  feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
  if feat is None:
    return val
  retTree = {}
  retTree['spInd'] = feat
  retTree['spVal'] = val
  lSet, rSet = binSplitDataSet(dataSet, feat, val)
  retTree['left'] = createTree(lSet, leafType, errType, ops)
  retTree['right'] = createTree(rSet, leafType, errType, ops)
  return retTree

def isTree(obj):
  return (type(obj).__name__ == 'dict')

def getMean(tree):
  if isTree(tree['right']):
    tree['right'] = getMean(tree['right']) 
  if isTree(tree['left']):
    tree['left'] = getMean(tree['left'])
  
  return (tree['left'] + tree['right'])/2.0

def prune(tree, testData):
  if np.shape(tree)[0] == 0:
    return getMean(testData)
  
  if isTree(tree['left']) or isTree(tree['right']):
    lSet, rSet = binSplitDataSet(tree, tree['spInd'], tree['spVal'])
  if isTree(tree['left']):
    tree['left'] = prune(lSet, tree['left'])
  if isTree(tree['right']):
    tree['right'] = prune(rSet, tree['right'])

  if not isTree(tree['left']) and not isTree(tree['right']):
    lSet, rSet = binSplitDataSet(tree, tree['spInd'], tree['spVal'])
    errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'], 2)) + np.sum(np.power(rSet[:,-1] - tree['right'], 2))
    treeMean = (tree['left'] + tree['right'])/2.0
    errorMerge = np.sum(np.power(testData[:,-1] - treeMean, 2))
    if errorMerge < errorNoMerge:
      print("merge")
      return treeMean
    else:
      return tree
  else:
    return tree

def modelLeaf(dataSet):
  ws, X, Y = linearSolve(dataSet)

def modelErr(dataSet):
  ws, X, Y = linearSolve(dataSet)
  yHat = X * ws
  return np.sum(np.power(Y - yHat, 2))

def linearSolve(dataSet):
  m,n = np.shape(dataSet)
  X = np.mat(np.ones((m, n)))
  Y = np.mat(np.ones((m, 1)))
  X[:,1:n] = dataSet[:,0:n-1]
  Y = dataSet[:,-1]
  xTx = X.T * Y
  if np.linalg.det(xTx) == 0.0:
      raise NameError('This matrix is singular')
  
  ws = xTx.I * (X.T * Y)
  return ws, X, Y

def regTreeEval(model, inData):
  return float(model)

def modelTreeEval(model, inData):
  n = np.shape(inData)[1]
  X = np.mat(np.ones((1, n)))
  X[:,1:n+1] = inData
  return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
  if not isTree(tree):
    return modelEval(tree, inData)
  
  if inData[0, tree['spInd']] <= tree['spVal']:
    if isTree(tree['left']):
      return treeForeCast(tree['left'], inData, modelEval)
    else:
      return modelEval(tree['left'], inData)
  else:
    if isTree(tree['right']):
      return treeForeCast(tree['right'], inData, modelEval)
    else:
      return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
      yHat[i,0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat

def loadDataSet(fileName):
    # assume last column is target value
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # map all elements to float()
        curLine = [float(value) for value in curLine]
        dataMat.append(curLine)
    return dataMat

if __name__ == "__main__":
#    testMat = np.mat(np.eye(4))
#    print(testMat)
#    print(type(testMat))
#    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
#    print(mat0, '\n-----------\n', mat1)

    # myDat = loadDataSet('data/9.RegTrees/data1.txt')
    # # myDat = loadDataSet('data/9.RegTrees/data2.txt')
    # # print 'myDat=', myDat
    # myMat = mat(myDat)
    # # print 'myMat=',  myMat
    # myTree = createTree(myMat)
    # print myTree

    # myDat = loadDataSet('data/9.RegTrees/data3.txt')
    # myMat = mat(myDat)
    # myTree = createTree(myMat, ops=(0, 1))
    # print myTree

    # myDatTest = loadDataSet('data/9.RegTrees/data3test.txt')
    # myMat2Test = mat(myDatTest)
    # myFinalTree = prune(myTree, myMat2Test)
    # print '\n\n\n-------------------'
    # print myFinalTree

    # # --------
    # myDat = loadDataSet('data/9.RegTrees/data4.txt')
    # myMat = mat(myDat)
    # myTree = createTree(myMat, modelLeaf, modelErr)
    # print myTree

     trainMat = np.mat(loadDataSet('bikeSpeedVsIq_train.txt'))
     testMat = np.mat(loadDataSet('bikeSpeedVsIq_test.txt'))
     myTree1 = createTree(trainMat, ops=(1, 20))
     print(myTree1)
     yHat1 = createForeCast(myTree1, testMat[:, 0])
     print ("--------------\n")
     print(yHat1)
     # print "ssss==>", testMat[:, 1]
     # print ("regTree:", np.corrcoef(yHat1, testMat[:, 1],rowvar=0)[0, 1])

    # myTree2 = createTree(trainMat, modelLeaf, modelErr, ops=(1, 20))
    # yHat2 = createForeCast(myTree2, testMat[:, 0], modelTreeEval)
    # print myTree2
    # print "modelTree:", corrcoef(yHat2, testMat[:, 1],rowvar=0)[0, 1]

    # ws, X, Y = linearSolve(trainMat)
    # print ws
    # m = len(testMat[:, 0])
    # yHat3 = mat(zeros((m, 1)))
    # for i in range(shape(testMat)[0]):
    #     yHat3[i] = testMat[i, 0]*ws[1, 0] + ws[0, 0]
    # print "lr:", corrcoef(yHat3, testMat[:, 1],rowvar=0)[0, 1]