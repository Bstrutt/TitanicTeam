import pandas as pd
import numpy as np
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

def status(feature):
    print('Processing', feature, ': ok')


def get_combined_data():
    # reading train data
    train = pd.read_csv("inputs/train.csv")

    # reading test data
    test = pd.read_csv("inputs/test.csv")

    # extracting and then removing the targets from the training data
    targets = train.Survived
    train.drop(['Survived'], 1, inplace=True)

    # merging train data and test data for future feature engineering
    # we'll also remove the PassengerID since this is not an informative feature
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop(['index'], inplace=True, axis=1)

    return combined


def recover_train_test_target():
    global combined

    targets = pd.read_csv('inputs/train.csv', usecols=['Survived'])['Survived'].values
    train = combined.iloc[:891]
    test = combined.iloc[891:]

    return train, test, targets


combined = get_combined_data()


def get_titles():
    global combined
    titles = set()
    for name in combined[:891]['Name']:
        titles.add(name.split(',')[1].split('.')[0].strip())

    title_dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Dona": "Mrs.",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"}

    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

    # a map of more aggregated title
    # we map each title
    combined['Title'] = combined.Title.map(title_dictionary)
    status('Title')
    return combined


def fill_age(row):
    global combined

    grouped_train = combined.iloc[:891].groupby(['Sex', 'Pclass', 'Title'])
    grouped_median_train = grouped_train.median()
    grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

    condition = (
            (grouped_median_train['Sex'] == row['Sex']) &
            (grouped_median_train['Title'] == row['Title']) &
            (grouped_median_train['Pclass'] == row['Pclass'])
    )
    return grouped_median_train[condition]['Age'].values[0]


def process_age():
    global combined
    # a function that fills the missing values of the Age variable
    combined['Age'] = combined.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    status('age')
    return combined


def process_names():
    global combined
    # we clean the Name variable
    combined.drop('Name', axis=1, inplace=True)

    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)

    # removing the title variable
    combined.drop('Title', axis=1, inplace=True)

    status('names')
    return combined


def process_fares():
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.iloc[:891].Fare.mean(), inplace=True)
    status('fare')
    return combined


def process_embarked():
    global combined
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    combined.Embarked.fillna('S', inplace=True)
    # dummy encoding
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', inplace=True, axis=1)
    status('embarked')
    return combined


def process_cabin():
    global combined

    # replacing missing cabins with U (for Unknown)
    combined.Cabin.fillna('U', inplace=True)

    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: str(c)[0])

    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    combined = pd.concat([combined, cabin_dummies], axis=1)

    combined.drop('Cabin', axis=1, inplace=True)
    status('cabin')
    return combined


def process_sex():
    global combined
    # mapping string values to numerical one
    combined['Sex'] = combined['Sex'].map({'male': 1, 'female': 0})
    status('Sex')
    return combined


def process_pclass():
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix="Pclass")

    # adding dummy variable
    combined = pd.concat([combined, pclass_dummies], axis=1)

    # removing "Pclass"
    combined.drop('Pclass', axis=1, inplace=True)

    status('Pclass')
    return combined


# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t: t.strip(), ticket)
    ticket = list(filter(lambda t: not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'XXX'


def process_ticket():
    global combined

    tickets = set()
    for t in combined['Ticket']:
        tickets.add(cleanTicket(t))

    # Extracting dummy variables from tickets.
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)

    status('Ticket')
    return combined


def process_family():
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    status('family')
    return combined

combined = get_titles()
combined = process_age()
combined = process_names()
combined = process_fares()
combined = process_embarked()
combined = process_cabin()
combined = process_sex()
combined = process_pclass()
combined = process_ticket()
combined = process_family()
train, test, targets = recover_train_test_target()


svm = svm.NuSVC(nu=0.5, gamma='auto', probability=True)
bc = BaggingClassifier(base_estimator=svm, n_estimators=20)

bc.fit(train, targets)
y = bc.predict(test)


if scale == 'standard':
    # Scale all values for easier model use
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

if model == 'svm':
    svm = svm.NuSVC(probability=True)
    bc = BaggingClassifier(base_estimator=svm, n_estimators=20)

if model == 'percep':
    bc = make_pipeline(Nystroem(gamma=0.2, n_components=300),
                   BaggingClassifier(base_estimator=SGDClassifier(loss='perceptron', max_iter=1000, tol=1e-3), n_estimators=20))

if model == 'rfc':
    bc = RandomForestClassifier(max_depth=8, random_state=0)

# Run selected model
bc.fit(train, targets)
y = bc.predict(test)

resultFile = open("outputs/result.csv", 'w')
resultFile.write("PassengerID,Survived\n")
for i in range(0, len(y)):
    resultFile.write(str(test.at[i, 'PassengerId']) + "," + str(y[i]) + "\n")
resultFile.close()
