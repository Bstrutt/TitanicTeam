import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

#Load training and test sets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#Set labels aside
trainSurvived = train['Survived']
#Set important features aside
trainFeatures = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin']


# Encode sex as 0 and 1 in both train and test
label_encoder = LabelEncoder()
train['Sex'] = label_encoder.fit_transform(train['Sex'])
test['Sex'] = label_encoder.transform(test['Sex'])

# Encode Cabin as binary 1=had cabin 0=no cabin
for i in range(0,len(train['Cabin'])):
    if str(train.at[i, 'Cabin']) == 'nan':
        train.at[i, 'Cabin'] = 0
    else:
        train.at[i, 'Cabin'] = 1

for i in range(0,len(test['Cabin'])):
    if str(test.at[i, 'Cabin']) == 'nan':
        test.at[i, 'Cabin'] = 0
    else:
        test.at[i, 'Cabin'] = 1

#impute missing age values
ageImputer = SimpleImputer()
train['Age'] = ageImputer.fit_transform(train['Age'].values.reshape(-1,1))
test['Age'] = ageImputer.transform(test['Age'].values.reshape(-1,1))

fareImputer = SimpleImputer()
test['Fare'] = fareImputer.fit_transform(test['Fare'].values.reshape(-1,1))

rfc = RandomForestClassifier(max_depth=8, random_state=0)
rfc.fit(train[trainFeatures],train['Survived'])

y = rfc.predict(test[trainFeatures])


resultFile = open("result.csv", 'w')
resultFile.write("PassengerID,Survived\n")
for i in range(0, len(y)):
    resultFile.write(str(test.at[i, 'PassengerId']) + "," + str(y[i]) + "\n")
resultFile.close()