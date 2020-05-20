import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# load training set into csv
with open("inputs/train.csv", 'r') as f:
    originalTrain = list(csv.reader(f, delimiter=","))
train = np.array(originalTrain[1:]) #train gets everything minus titles

#load testing set into csv
with open("inputs/test.csv", 'r') as f1:
    originalTest = list(csv.reader(f1, delimiter=","))
test = np.array(originalTest[1:]) # test gets everything minus titles

# Creating our training set outcomes
survived = np.empty([len(train)], dtype=int)
for i in range(0, len(train)):
    survived[i] = train[i][1]

# Grabbing a passenger Ids of the test set
testIds = np.empty([len(test)], dtype=int)
for i in range(0, len(test)):
    testIds[i] = test[i][0]


# clean training set
for i in range(0, len(train)):
    if train[i][10] == '':
        train[i][10] = 0
    else:
        train[i][10] = 1
    if train[i][5] == '':
        train[i][5] = 0
    if train[i][4] == 'male':
        train[i][4] = 0
    else:
        train[i][4] = 1

# clean test set
for i in range(0, len(test)):
    if test[i][9] == '':
        test[i][9] = 0
    else:
        test[i][9] = 1

    if test[i][4] == '':
        test[i][4] = 0

    if test[i][3] == 'male':
        test[i][3] = 0
    else:
        test[i][3] = 1

    if test[i][8] =='':
        test[i][8] = 0

train = np.delete(train, [0,1,3,8,11],1) # remove
test = np.delete(test, [0,2,7,10],1)
train = train.astype('float32')
test = test.astype('float32')

rfc = RandomForestClassifier(max_depth=8, random_state=0)
rfc.fit(train,survived)

testPredictions = rfc.predict(test)

resultFile = open("outputs/result.csv", 'w')
resultFile.write("PassengerID,Survived\n")
for i in range(0, len(testPredictions)):
    resultFile.write(str(testIds[i]) + "," + str(testPredictions[0]) + "\n")
resultFile.close()
