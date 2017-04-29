#!/usr/bin/python
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

# Task 1: Select what features you'll use.
features_list = ['poi',
                 'salary', "to_messages",
                 "total_payments",
                 "exercised_stock_options",
                 "bonus",
                 "restricted_stock",
                 "shared_receipt_with_poi",
                 "restricted_stock_deferred",
                 "total_stock_value",
                 "expenses",
                 "loan_advances",
                 "from_messages",
                 "from_this_person_to_poi",
                 "director_fees",
                 "deferred_income",
                 "long_term_incentive",
                 "from_poi_to_this_person"]

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# load data into a dataframe
df = pd.DataFrame.from_dict(data_dict, orient='index')
# replace the NaN for our exploration - undo it for the machine learning part

df = df.replace('NaN', np.nan)
df = df[features_list]
df.info()
features_list.remove('restricted_stock_deferred')
features_list.remove('director_fees')
features_list.remove('loan_advances')

df = df.drop('restricted_stock_deferred', 1)
df = df.drop('director_fees', 1)
df = df.drop('loan_advances', 1)

print len(features_list)

# Task 2: Remove outliers
df[df.isnull().sum(axis=1) >= 15]

money_df = pd.DataFrame({"Salaray": df['salary'],
                         "Bonus": df['bonus']})
print money_df.describe()

new = df.filter(['salary', 'bonus'], axis=1)

print new.nlargest(5, 'salary')
print new.nlargest(10, 'bonus')

df.plot.scatter('salary', 'bonus')

df = df.drop('TOTAL')
df = df.drop('LOCKHART EUGENE E')
df = df.drop('THE TRAVEL AGENCY IN THE PARK')

data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


# Task 3: Create new feature(s)
for k in data_dict.keys():
    data_dict[k]['ratio_from_poi'] = 0
    if (data_dict[k]['from_poi_to_this_person'] != 'NaN') and (data_dict[k]['from_messages'] != 'NaN') and (data_dict[k]['from_messages'] != 0):
        data_dict[k]['ratio_from_poi'] = float(
            data_dict[k]['from_poi_to_this_person']) / float(data_dict[k]['from_messages'])
features_list.append('ratio_from_poi')

for k in data_dict.keys():
    data_dict[k]['ratio_to_poi'] = 0
    if (data_dict[k]['from_this_person_to_poi'] != 'NaN') and (data_dict[k]['to_messages'] != 'NaN') and (data_dict[k]['to_messages'] != 0):
        data_dict[k]['ratio_to_poi'] = float(
            data_dict[k]['from_this_person_to_poi']) / float(data_dict[k]['to_messages'])
features_list.append('ratio_to_poi')

for k in data_dict.keys():
    data_dict[k]['ratio_bonus'] = 0
    if (data_dict[k]['total_payments'] != 'NaN') and (data_dict[k]['bonus'] != 'NaN') and (data_dict[k]['bonus'] != 0):
        data_dict[k]['ratio_bonus'] = float(
            data_dict[k]['total_payments']) / float(data_dict[k]['bonus'])
features_list.append('ratio_bonus')

# update feature_list
features_list.append('ratio_from_poi')
features_list.append('ratio_to_poi')
features_list.append('ratio_bonus')

# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Task 4: Try a varity of classifiers
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

k_best = SelectKBest(k=7)
k_best.fit(features, labels)
scores = k_best.scores_
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))

print 'Select KBest', sorted_pairs

feature_list = features_list[1:8]

# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

test_classifier(clf, my_dataset, features_list)

from sklearn import svm

svc = SVC(kernel="linear")
estimators = [('scale', StandardScaler()), ('svc', svc)]
clf = Pipeline(estimators)

test_classifier(clf, my_dataset, features_list)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

knn = KNeighborsClassifier()
estimators = [('scale', StandardScaler()), ('knn', knn)]
clf = Pipeline(estimators)
test_classifier(clf, my_dataset, features_list)


# Task 5: Tune your classifier to achieve better than .3 precision and recall
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}

clf = GridSearchCV(SVC(kernel='linear', class_weight='balanced'), param_grid)
test_classifier(clf, my_dataset, features_list)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
