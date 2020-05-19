import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


model = 'svm'
scale = 'standard'

# Load training and test sets
train = pd.read_csv("inputs/train.csv")
test = pd.read_csv("inputs/test.csv")

# Set labels aside
trainSurvived = train['Survived']
# Set important features aside
trainFeatures = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']


# Fill Embarked values
for i in range(0, len(train['Embarked'])):
    if str(train.at[i, 'Embarked']) == 'nan':
        train.at[i, 'Embarked'] = 'S'

# Encode emarking port
embark_encoder = LabelEncoder()
train['Embarked'] = embark_encoder.fit_transform(train['Embarked'])
test['Embarked'] = embark_encoder.fit_transform(test['Embarked'])

# Encode sex as 0 and 1 in both train and test
label_encoder = LabelEncoder()
train['Sex'] = label_encoder.fit_transform(train['Sex'])
test['Sex'] = label_encoder.transform(test['Sex'])



# Encode Cabin as binary 1=had cabin 0=no cabin
for i in range(0, len(train['Cabin'])):
    if str(train.at[i, 'Cabin']) == 'nan':
        train.at[i, 'Cabin'] = 0
    else:
        train.at[i, 'Cabin'] = 1

for i in range(0, len(test['Cabin'])):
    if str(test.at[i, 'Cabin']) == 'nan':
        test.at[i, 'Cabin'] = 0
    else:
        test.at[i, 'Cabin'] = 1

# impute missing age values
imp = IterativeImputer(max_iter=10, random_state=0)
train[trainFeatures] = imp.fit_transform(train[trainFeatures])
test[trainFeatures] = imp.transform(test[trainFeatures])

if scale == 'standard':
    # Scale all values for easier model use
    scaler = StandardScaler()
    train[trainFeatures] = scaler.fit_transform(train[trainFeatures])
    test[trainFeatures] = scaler.transform(test[trainFeatures])


if model == 'svm':
    svm = svm.NuSVC(probability=True)
    bc = BaggingClassifier(base_estimator=svm, n_estimators=20)

if model == 'percep':
    bc = make_pipeline(Nystroem(gamma=0.2, n_components=300),
                   BaggingClassifier(base_estimator=SGDClassifier(loss='perceptron', max_iter=1000, tol=1e-3), n_estimators=20))

if model == 'rfc':
    bc = RandomForestClassifier(max_depth=8, random_state=0)

# Run selected model
bc.fit(train[trainFeatures], train['Survived'])
y = bc.predict(test[trainFeatures])

resultFile = open("outputs/result.csv", 'w')
resultFile.write("PassengerID,Survived\n")
for i in range(0, len(y)):
    resultFile.write(str(test.at[i, 'PassengerId']) + "," + str(y[i]) + "\n")
resultFile.close()
