# Jake Krayger
# k-Nearest Neighbor
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.datasets import load_iris


k = 7

iris = load_iris()
data = np.array(iris["data"])
labels = np.array(iris["target"])

# 2D
# data = [[1, 1], [2, 1], [1.5, 2], [2, 2], [1, 3],
# [3, 1], [1.5, 1.5], [2.5, 2], [1.8, 2.8], [1, 2.5],
# [4, 4], [5, 5], [4.5, 5], [5, 4.5], [6, 6], [3.5, 4.2],
# [4.8, 4.8], [5.2, 5.3], [6.5, 6.5], [5, 6]]

# 3D
# data = [[1, 1, 1], [2, 1, 1], [1.5, 2, 1], [2, 2, 1], [1, 3, 1],
# [3, 1, 1], [1.5, 1.5, 1], [2.5, 2, 1], [1.8, 2.8, 1], [1, 2.5, 1],
# [4, 4, 6.5], [5, 5, 6.5], [4.5, 5, 6.5], [5, 4.5, 6.5], [3.5, 3.5, 3.5],
# [3.5, 4.2, 6.5], [4.8, 4.8, 6.5], [5.2, 5.3, 6.5], [3, 6.5, 6.5], [5, 6, 6.5]]

# labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

test_point = [7, 4, 5.99, 6]

def splitData(d):
    points = []
    labels = []
    for i in d:
        points.append(i[0])
        labels.append(i[1])
    
    return (points, labels)

#
def scale(data, test_point):
    maxByFeat = [max([j[z] for j in [i for i in data]]) for z in range(len(data[0]))]
    scaled_data = []
    scaled_testPoint = []

    for m in data:
        temp = []
        for t in range(len(maxByFeat)):
            temp.append(m[t] / maxByFeat[t])
        scaled_data.append(np.array(temp))
    
    for b in range(len(maxByFeat)):
        scaled_testPoint.append(test_point[b] / maxByFeat[b])

    return (scaled_data, scaled_testPoint)


# euclidean distance
def euc(point1, point2):
    dis = 0
    for i in range(len(point1)):
        dis += (point2[i] - point1[i])**2
    return math.sqrt(dis)


# need to break ties and clean this
def nearestNeighbor(k, point, data):
    dists = []
    for i in data:
        d = euc(point, i[0])
        dists.append((i, d))
    
    dists = sorted(dists, key = lambda x: x[1])[:k]
    classes = [j[0][1] for j in dists]
    print(classes)

    counts = {}

    for z in classes:
        counts[z] = counts.get(z, 0) + 1
    

    mex = 0
    max_ = 4
    for l in counts:
        if counts.get(l) > mex:
            mex = counts.get(l)
            max_ = l
    
    return (point, max_)


def plot(d):
    t = np.array([i for i in d[0]])
    if dimensions == 3:
        t = np.multiply(t, 125)
        print(t)
        plt.scatter(t[:, 0], t[:, 1], t[:, 2], c=d[1], alpha = 0.5, edgecolors='black')
        plt.scatter(t[len(t) - 1][0], t[len(t) - 1][1], c='red', marker='x')
    else:
        plt.scatter(t[:, 0], t[:, 1], c=d[1], alpha = 0.5, edgecolors='black')
        plt.scatter(t[len(t) - 1][0], t[len(t) - 1][1], c='red', marker='x')

    plt.show()


scale_data = scale(data, test_point)
train_data = list(zip(scale_data[0], labels))

nn = nearestNeighbor(k, scale_data[1], train_data)
add_pred = train_data.copy()
add_pred.append(nn)

dimensions = len(scale_data[1])
plot(splitData(add_pred))