# coding=utf-8
from numpy import *
import math
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iterate = 500

def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float, curLine)
		dataMat.append(fltLine)
	return dataMat

# 计算两个样本欧几里得距离
def euclindeanDistance(instance1, instance2, col):
	distance = 0
	for x in range(col):
		distance += pow(instance1[:, x] - instance2[:, x], 2)
	return math.sqrt(distance)

# 随机生成初始的质心
def initCentroids(dataSet, k):
	numSamples, dim = dataSet.shape
	centroids = mat(zeros((k, dim)))
	index = random.sample(range(numSamples), k)
	for i in range(k):
		centroids[i, :] = dataSet[index[i], :]
	return centroids


def kMeans(dataSet, k, distMeas= euclindeanDistance, createCent=initCentroids):
	row, col = dataSet.shape

	clusterAssment = mat(zeros((row, col)))  # create mat to assign data points
	# to a centroid, also holds SE of each point
	centroids = createCent(dataSet, k)
	clusterChanged = True
	iter = 0
	while clusterChanged & (iter < iterate):
		clusterChanged = False
		iter = iter + 1
		for i in range(row):  # for each data point assign it to the closest centroid
			minDist = inf
			minIndex = -1
			tempDist = list(zeros(k))
			for j in range(k):
				tempDist[j] = euclindeanDistance(centroids[j, :], dataSet[i, :], col)
			minDist = min(tempDist)
			minIndex = tempDist.index(minDist)

			if clusterAssment[i, 0] != minIndex:
				clusterChanged = True
			clusterAssment[i, :] = minIndex, minDist ** 2
		print(centroids)

		for cent in range(k):  # recalculate centroids
			ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
			centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean
	return centroids, clusterAssment


def main():
	## step 1: load data
	print
	"step 1: load data..."
	dataSet = []
	fileIn = open('C:/Users/Graeme/PycharmProjects/typhoon/testdata.txt')
	for line in fileIn.readlines():
		lineArr = line.strip().split('\t')
		dataSet.append([float(lineArr[0]), float(lineArr[1])])

	## step 2: clustering...
	print
	"step 2: clustering..."
	dataSet = mat(dataSet)
	k = 4
	centroids, clusterAssment = kMeans(dataSet, k)

	## step 3: show the result
	print
	"step 3: show the result..."
	plt.scatter(array(dataSet[:, 0]), array(dataSet[:, 1]), c=array(clusterAssment[:, 0]))
	plt.show()

if __name__ == '__main__':
	main()
