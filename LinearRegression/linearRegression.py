import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(filename):
  dataMat = []
  labelMat = []
  with open(filename) as data_file:
    for line in data_file:
      lineArr = line.strip().split('\t')
      dataMat.append( [float(value) for value in lineArr[:-1]] )
      labelMat.append( float(lineArr[-1]) )
  return dataMat, labelMat

def standRegres(samples, labels):
  sampleMat = np.mat(samples)
  labelMat = np.mat(labels).T
  xTx = sampleMat.T * sampleMat
  if np.linalg.det(xTx) == 0.0:
    print("This matrix is singular, cannot do invers")
    return
  
  return xTx.I * sampleMat.T * labelMat

def regression1():
  samples,labels = loadDataSet("ex0.txt");
  w = standRegres(samples, labels)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter([value[1] for value in samples], labels)
  samplesCopy = np.mat(samples).copy()
  samplesCopy.sort(0)
  y = samplesCopy*w
  ax.plot(samplesCopy[:,1], y)
  plt.show()

def lwlr(testPoint, xArr, yArr, k=1.0):
  xMat = np.mat(xArr)
  yMat = np.mat(yArr).T
  m = np.shape(xMat)[0]
  weights = np.mat(np.eye((m)))
  for i in range(m):
    diffMat = testPoint - xMat[i,:]
    weights[i,i] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
#    print(weights[i,i])
  xTx = xMat.T * (weights*xMat)
  if np.linalg.det(xTx) == 0.0:
    print("This matrix is singular, cannot do invers")
    return
  
  ws = xTx.I * (xMat.T * weights * yMat)
  return testPoint*ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
  m = np.shape(testArr)[0]
  yHat = np.zeros(m)
  for i in range(m):
    yHat[i] = lwlr(testArr[i], xArr, yArr, k)
  return yHat

def regression2():
    xArr, yArr = loadDataSet("ex0.txt")
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = np.mat(xArr)
    srtInd = xMat[:,1].argsort(0)           # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出
    xSort=xMat[srtInd][:,0,:]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1], yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0], np.mat(yArr).T.flatten().A[0] , s=2, c='red')
    plt.show()
  
def rssError(yArr, yHatArr):
  return ((yArr-yHatArr)**2).sum()

def alaloneTest():
  xArr, yArr = loadDataSet("abalone.txt")
  yHat01 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 0.1)
  print("yHat01 error size is : ", rssError(yArr[0:99], yHat01.T))
  yHat1 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 1)
  print("yHat11 error size is : ", rssError(yArr[0:99], yHat1.T))
  yHat10 = lwlrTest(xArr[0:99], xArr[0:99], yArr[0:99], 10)
  print("yHat10 error size is : ", rssError(yArr[0:99], yHat10.T))

  newHat01 = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 0.1)
  print("new yHat01 error size is : ", rssError(yArr[0:99], newHat01.T))
  newHat1 = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 1)
  print("new yHat1 error size is : ", rssError(yArr[0:99], newHat1.T))
  newHat10 = lwlrTest(xArr[100:199], xArr[0:99], yArr[0:99], 10)
  print("new yHat10 error size is : ", rssError(yArr[0:99], newHat10.T))

  standWs = standRegres(xArr[0:99], yArr[0:99])
  standHat = np.mat(xArr[100:199])*standWs
  print("new standHat error size is : ", rssError(yArr[0:99], standHat.T.A))

alaloneTest();
#regression1()
#regression2()