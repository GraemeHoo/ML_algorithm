import numpy as np
import pandas as pd
import math
import os

def files_abspath():
	srcPath = r"weather.arff"
	path = os.path.abspath(srcPath)
	return path

# 读取arff文件
def read_arff(filename):
    list = []
    arff_file = open(filename)
    for line in arff_file:
        if not (line.startswith("@")):
            if not (line.startswith("%")):
                line = line.strip("\n")
                line = line.split(',')
                list.append(line)
    while [''] in list:
        list.remove([''])
    arr = np.array(list)
    return arr

# 将数据划分为训练集、测试集
def split_data(data, test_size):
	if(test_size != 0):
		data_num = data.shape[0]
		train_index = list(range(data_num))
		test_index = []
		test_num = int(data_num * test_size)

		for i in range(test_num):
			random_index = int(np.random.uniform(0, len(train_index)))
			test_index.append(train_index[random_index])
			del train_index[random_index]
		train = data[train_index]
		test = data[test_index]
	else:
		train = data
		test = data
	print('The percentage of test data is:', test_size)
	return train, test

# 划分不同类别数据
def separate_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

# 假设连续型数据服从高斯分布，且相互独立
def gaussian(x, mu, sigma):
    val = 1 / math.sqrt(2 * math.pi * sigma)
    return val * (math.exp((-(x - mu) ** 2) / (2 * sigma**2)))

# 计算条件概率
def calculate_condi_prob(seperated_data, alpha):
    df = pd.DataFrame(seperated_data, columns=['outlook', 'temperature', 'humidity', 'windy', 'PlayORNot'])

    total_case = df.shape[0]

    sunny = df['outlook'].tolist().count('sunny')
    rainy = df['outlook'].tolist().count('rainy')
    overcast = df['outlook'].tolist().count('overcast')

    sunny_prob = (sunny + alpha) / (total_case + 3 * alpha)
    rainy_prob = (rainy + alpha) / (total_case + 3 * alpha)
    overcast_prob = (overcast + alpha) / (total_case + 3 * alpha)

    wind = df['windy'].tolist().count('TRUE')
    nowind = df['windy'].tolist().count('FALSE')

    wind_prob = (wind + alpha) / (total_case + 2 * alpha)
    nowind_prob = (nowind + alpha) / (total_case + 2 * alpha)

    df['temperature'] = df['temperature'].astype('int')
    temper_mean = df['temperature'].mean()
    temper_std = df['temperature'].std()

    df['humidity'] = df['humidity'].astype('int')
    humidity_mean = df['humidity'].mean()
    humidity_std = df['humidity'].std()

    return sunny_prob, rainy_prob, overcast_prob, wind_prob, nowind_prob,\
           temper_mean, temper_std, humidity_mean, humidity_std


# 概率计算
def calculate_prob(traindata, alpha):
    sep_data = separate_class(traindata)

    class_yes_data = np.vstack(sep_data['yes'])
    class_no_data = np.vstack(sep_data['no'])

    # 计算两类先验概率
    prior_prob0 = (len(class_no_data) + alpha) / (len(class_yes_data)+len(class_no_data)+2*alpha)  # class 0:no prior prob
    prior_prob1 = (len(class_yes_data) + alpha) / (len(class_yes_data)+len(class_no_data)+2*alpha)  # class 1:yes prior prob

    # 计算条件概率
    condi_prob0 = calculate_condi_prob(class_no_data, alpha)
    condi_prob1 = calculate_condi_prob(class_yes_data, alpha)

    return (prior_prob0, prior_prob1), condi_prob0, condi_prob1   # 返回概率集合


def bayes_classifer(traindata, testdata, alpha):
	'''
	condi_prob return sunny_prob, rainy_prob, overcast_prob, wind_prob, nowind_prob,\
						temper_mean, temper_std, humidity_mean, humidity_std

	bayes_classifer return (prior_prob0, prior_prob1), condi_prob0, condi_prob1

	alpha = 1 is Laplace transform coefficient
	'''
	row = np.shape(testdata)[0]

	probset = calculate_prob(traindata, alpha)
	temper_data = testdata[:, 1].astype(int)
	humi_data = testdata[:, 2].astype(int)

	testdata_class = []
	for i in range(len(testdata)):
		if testdata[i][0] == 'sunny':
			outlook_prob = probset[1][0], probset[2][0]
		elif testdata[i][0] == 'rainy':
			outlook_prob = probset[1][1], probset[2][1]
		else:
			outlook_prob = probset[1][2], probset[2][2]

		if testdata[i][3] == 'TURE':
			wind_prob = probset[1][3], probset[2][3]
		else:
			wind_prob = probset[1][4], probset[2][4]

		temper_prob = gaussian(temper_data[i], probset[1][5], probset[1][6]), \
					  gaussian(temper_data[i], probset[2][5], probset[2][6])

		humi_prob = gaussian(humi_data[i], probset[1][7], probset[1][8]), \
					gaussian(humi_data[i], probset[2][5], probset[2][6])

		class_prob = []
		for j in range(2):
			class_prob.append(probset[0][j]*outlook_prob[j]*wind_prob[j]*temper_prob[j]*humi_prob[j])
		if class_prob[0] < class_prob[1]:
			testdata_class.append('yes')
		elif class_prob[0] > class_prob[1]:
			testdata_class.append('no')

	ground_true = testdata[:, 4]
	count = 0
	for k in range(row):
		if (ground_true[k] == testdata_class[k]):
			count += 1

	accuracy = count / row
	print('The true label:\n', ground_true)
	print('The model result:\n', testdata_class)
	print('Appear', row-count, 'mistake and the accuracy is :', accuracy)
	return accuracy


if __name__ == '__main__':
	arr = read_arff(files_abspath())
	trainSets, testSets = split_data(arr, test_size=0) # 30% test data, 70% training data
	bayes_classifer(trainSets, testSets, alpha=0)
