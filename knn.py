# Jake Krayger
# k-Nearest Neighbor
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.datasets import load_iris


k = 10
dimensions = 3

iris = load_iris()
data = np.array(iris["data"])

# 2D
# data = [([1, 1], 0), ([2, 1], 0), ([1.5, 2], 0), ([2, 2], 0), ([1, 3], 0),
# ([3, 1], 0), ([1.5, 1.5], 0), ([2.5, 2], 0), ([1.8, 2.8], 0), ([1, 2.5], 0),
# ([4, 4], 1), ([5, 5], 1), ([4.5, 5], 1), ([5, 4.5], 1), ([6, 6], 1),
# ([3.5, 4.2], 1), ([4.8, 4.8], 1), ([5.2, 5.3], 1), ([6.5, 6.5], 1), ([5, 6], 1)]

# 3D
data = [([1, 1, 1], 0), ([2, 1, 1], 0), ([1.5, 2, 1], 0), ([2, 2, 1], 0), ([1, 3, 1], 0),
([3, 1, 1], 0), ([1.5, 1.5, 1], 0), ([2.5, 2, 1], 0), ([1.8, 2.8, 1], 0), ([1, 2.5, 1], 0),
([4, 4, 6.5], 1), ([5, 5, 6.5], 1), ([4.5, 5, 6.5], 1), ([5, 4.5, 6.5], 1), ([6, 6, 6.5], 1),
([3.5, 4.2, 6.5], 1), ([4.8, 4.8, 6.5], 1), ([5.2, 5.3, 6.5], 1), ([6.5, 6.5, 6.5], 1), ([5, 6, 6.5], 1)]

test_point = [3.2, 3.2, 3.5]

points = []
labels = []

def splitData(d):
    for i in d:
        points.append(i[0])
        labels.append(i[1])

# need to scale by feature not just greatest value
def scale(d):
    return [i / max(np.array(points).flatten()) for i in [j for j in d]]


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
    max_ = 0
    for l in counts:
        if counts.get(l) > mex:
            mex = counts.get(l)
            max_ = l
    
    return (point, max_)


def plot():
    t = np.array([i[0] for i in scale_data])
    if dimensions == 3:
        t = np.multiply(t, 125)
        plt.scatter(t[:, 0], t[:, 1], t[:, 2], c=labels, alpha = 0.5, edgecolors='black')
        plt.scatter(t[len(t) - 1][0], t[len(t) - 1][1], c='red', marker='x')
    else:
        plt.scatter(t[:, 0], t[:, 1], c=labels, alpha = 0.5, edgecolors='black')
        plt.scatter(t[len(t) - 1][0], t[len(t) - 1][1], c='red', marker='x')

    plt.show()


nn = nearestNeighbor(5, test_point, data)
add_pred = data.copy()
add_pred.append(nn)

splitData(add_pred)
scale_data = list(zip(scale(points), labels))

plot()