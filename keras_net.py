import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

from matplotlib import pyplot as plt
import csv
from itertools import islice


K = 15

Learn_size = 210
test_size = 40
predict_size = 40

data = []
num_examples = 0
ile = 0

X_train = []
y_train = []

X_test = []
y_test = []

licznik = 0


# load train data
for i in range(K):
    points = []
    with open('dane/dane%d.txt' %i) as file:
        reader = csv.reader(file, delimiter=',', skipinitialspace=True)
        for row in islice(reader,0,Learn_size):
            X_train.append([float(row[0]), float(row[1])])
            y_train.append(i)
            licznik = licznik + 1
    data.append(np.asarray(points))

# load test data

for i in range(K):
    with open('dane/dane%d.txt' %i) as file:
        reader = csv.reader(file, delimiter=',', skipinitialspace=True)
        for row in islice(reader,Learn_size,Learn_size + test_size):
            X_test.append([float(row[0]), float(row[1])])
            y_test.append(i)
            licznik = licznik + 1

X = []
Y = []

# load data to predict
for i in range(K):
    with open('dane/dane%d.txt' %i) as file:
        reader = csv.reader(file, delimiter=',', skipinitialspace=True)
        for row in islice(reader,Learn_size + test_size ,Learn_size + test_size +  predict_size ):
            X.append([float(row[0]), float(row[1])])
            licznik = licznik + 1


X_train /= np.std(X_train, axis = 0)
X_test /= np.std(X_test, axis = 0)
X /= np.std(X, axis = 0)

X_test = np.asarray(X_test)
X_train = np.asarray(X_train)
X = np.asarray(X)

y_test = np.asarray(y_test)
y_train = np.asarray(y_train)

model = Sequential()

model.add(Dense(32, activation='relu', input_dim=2))
model.add(Dense(64, activation='relu'))
model.add(Dense(K, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


Y_train = keras.utils.to_categorical(y_train, K)
Y_test = keras.utils.to_categorical(y_test, K)

model.fit(X_train, Y_train, epochs=16, batch_size=16)

score = model.evaluate(X_test,Y_test, batch_size=32)

print(score)

separated = []

for i in range(K * predict_size):
    value = np.array(X[i]).reshape((1, 2))
    result = model.predict(value)
    predicted_cluster = np.argmax(result)
    temp = [predicted_cluster,X[i]]
    separated.append(temp)

#print(separated)
new_group = []
all = []

for k in range(K):
    new_group = []
    zliczanie = 0
    zaden = True
    for i in range(K * predict_size):
        if separated[i][0] == k:
            new_group.append(X[i])
            zliczanie = zliczanie + 1
            zaden = False
    print(" DLa klastra :", k, " znalazlem :", zliczanie, " punktow.")
    if zaden:
        new_group.append([0,0])
    all.append(np.asarray(new_group))

all = np.asarray(all)

#print(all)

for i in range(K):
    plt.plot(all[i][:][:,0],all[i][:][:,1], '.')

plt.show()
