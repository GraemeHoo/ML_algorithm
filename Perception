'''
perception algorithm by Graeme 2017/8/3
dataset (two dimension)
'''

import numpy as np
import matplotlib.pyplot as plt
# train_data [ x1,x2,Y ]

def MLP(train_data, l_rate):

    weights = [1.0 for i in range(len(train_data[0]) - 1)]
    bias = 0.0
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    column1 = [t[0] for t in train_data]
    column2 = [t[1] for t in train_data]
    ax1.scatter(column1, column2)
    ax1.axis([0,max(column1)+0.2, 0, max(column2)+0.2])
    plt.ion()
    plt.show()

    inter = 1
    flag = 0
    record = 0
    nochange_upper_limit = len([t[0] for t in train_data])

    while flag == 0:
        count = 0
        for index in range(len([t[0] for t in train_data])):
            temp = train_data[index][-1] * (np.matmul(weights, train_data[index][0:len(train_data[0])-1])+bias)
            print('temp:', temp)
            if temp <= 0:
                weights += np.dot(train_data[index][-1] * l_rate, train_data[index][0:len(train_data[0])-1])
                bias += l_rate * train_data[index][-1]
                record = ([t[-1] for t in train_data]) * (np.matmul(weights, np.transpose(np.column_stack([column1,column2]))) + bias)
                error = [i for i in record if i <= 0]
                ax2.scatter(inter, -np.sum(error))
                inter = inter + 1
                ax1.plot([0, -bias/weights[1]], [-bias/weights[0], 0])
                plt.pause(1)
                break
            else:
                count += 1
                print('count:', count)
                if count >= nochange_upper_limit:
                    flag = 1
    print(weights)
    print(bias)
    print(record)
    return weights, bias

# Create the dataset
x1 = np.round(np.random.normal(0.3, 0.1, 15), 2)
y1 = np.round(np.random.normal(0.3, 0.1, 15), 2)
x2 = np.round(np.random.normal(1, 0.1, 15), 2)
y2 = np.round(np.random.normal(1, 0.1, 15), 2)
label_size = len(x1)
data1 = np.column_stack([x1, y1, np.ones(label_size, dtype=int)])
data2 = np.column_stack([x2, y2, -np.ones(label_size, dtype=int)])
data = np.vstack((data1, data2))

MLP(data, 0.1)
