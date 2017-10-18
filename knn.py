import matplotlib.pyplot as plt
from numpy import *

def euclindeanDistance(instance1, instance2):
	col = instance1.shape[1]
	distance = 0
	for x in range(col):
		distance += pow(instance1[:, x] - instance2[:, x], 2)
	return math.sqrt(distance)


def KNN(dataSet, new_data, k):  # new_data 1*n
	row, col = dataSet.shape
	nearbor_distance = mat(zeros((row, 2)))
	for index in range(len(dataSet)):
		nearbor_distance[index, :] = index, euclindeanDistance(dataSet[index, :], new_data)

	kneardistance = array(nearbor_distance)
	new_kneardistance = kneardistance[lexsort(kneardistance.T)]
	return new_kneardistance[0:k, :]


def main():
	data = ([1, 1], [4, 4], [1, 1.5], [3, 4], [2.1, 2])
	new_data = ([2, 2])
	dataMat = mat(data)
	new_dataMat = mat(new_data)
	distance_matrix = KNN(dataMat, new_dataMat, 2)
	print(distance_matrix)
	plt.scatter(array(dataMat[:, 0]), array(dataMat[:, 1]))
	plt.scatter(array(new_dataMat[:, 0]), array(new_dataMat[: ,1]))
	plt.show()

if __name__ == '__main__':
    main()
