import csv
from matplotlib import pyplot as plt
import numpy as np
import math
from random import randrange


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def separate_points(X, Y, centroids):
    groups = []

    for counter, x in enumerate(X):
        min_distance = math.inf
        group = -1

        for i in range(K):
            dist = euclidean_distance(x, Y[counter], centroids[i,0], centroids[i,1])
            if dist < min_distance:
                group = i
                min_distance = dist
        groups.append(group)

    separated = []

    for i in range(K):
        new_group = []
        for counter, group in enumerate(groups):
            if group == i:
                new_group.append([X[counter], Y[counter]])

        new_group = np.asarray(new_group)
        separated.append(new_group)

    separated = np.asarray(separated)
    return separated

def compute_centroids(separated,centroids,end):

    for i in range(K):
        x_average = 0
        y_average = 0

        for counter, x in enumerate(separated[:][i][:, 0]):
            x_average += x
            y_average += separated[:][i][:, 1][counter]

        size = len(separated[:][i][:, 0])

        x_average /= size
        y_average /= size

        if centroids[i][0] == x_average and centroids[i][1] == y_average:
            end = False
        else:
            end = True
            centroids[i][0] = x_average
            centroids[i][1] = y_average

    return centroids, end


if __name__ == "__main__":

    Points = []

    with open('breast.txt') as file:
        reader = csv.reader(file, delimiter=' ', skipinitialspace=True)
        for row in reader:
            Points.append([float(row[0]), float(row[1])])

    Points = np.asarray(Points)
    # x values of points
    X = Points[:, 0]
    # y value of points
    Y = Points[:, 1]

    x_min = min(X)
    x_max = max(X)
    y_min = min(Y)
    y_max = max(Y)

    K = 15  # int(input("Podaj ilosc centroidów: "))

    centroids = []

    random_indexes = np.linspace(0, len(X) - 1, K)

    #random_indexes = np.random.uniform(0,len(X) - 1, K)


    for i in range(K):
        random_index = int(random_indexes[i])
        x = Points[random_index, 0]
        y = Points[random_index, 1]
        centroids.append([x, y])

    centroids = np.asarray(centroids)

    plt.plot(Points[:, 0], Points[:, 1], ".")
    plt.plot(centroids[:, 0], centroids[:, 1], 'x')
    plt.show()

    separated = separate_points(X, Y, centroids)
    plt.axis("equal")

    # jak nie przyporzadkuje do jakiegos centroidu punktow to wtedy wychodzi poza zakres tablicy
    for i in range(K):
        print(i)
        plt.plot(separated[i][:, 0], separated[i][:, 1], 'o')
        plt.plot(centroids[i, 0], centroids[i, 1], 'x')

    plt.show()

    end = True
    while end:
        centroids, end = compute_centroids(separated,centroids,end)
        separated = separate_points(X, Y, centroids)

    for i in range(K):
        plt.plot(separated[:][i][:, 0], separated[:][i][:, 1], '.')
        plt.plot(centroids[:, 0], centroids[:, 1], 'x')
        np.savetxt('dane/dane%d.txt' % i , separated[:][i], delimiter=',')



    plt.show()
