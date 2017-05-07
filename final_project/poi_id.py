#!/usr/bin/python

from __future__ import print_function
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

from amount_of_features import calc_for_features_list_length
# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

first_data_item = [v for v in data_dict.values()[:1]]
print ('Data Example: ', (json.dumps(first_data_item, indent=2)))

data_points = len(data_dict)
print ('-----------------------------------------')
print ('Total number of data points:', data_points)

# add all features first to list and check the NaNs
# 'poi' has to be first, because it is the target
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


# load data into a dataframe
df = pd.DataFrame.from_dict(data_dict, orient='index')
# replace the NaN for our exploration - undo it for the machine learning part
df = df.replace('NaN', np.nan)
df = df[features_list]
df.info()

poi_to_non_poi = df.poi.value_counts()
poi_to_non_poi.index = ['non persons of interest', 'persons of interest']

print (poi_to_non_poi)

# Task 2: Remove outliers
df[df.isnull().sum(axis=1) >= 16]
df = df.drop('TOTAL')
df = df.drop('LOCKHART EUGENE E')
df = df.drop('THE TRAVEL AGENCY IN THE PARK')

data_dict.pop('TOTAL', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)


# data_dict.pop("TOTAL", 0)
# df = df.drop('TOTAL')
money_df = pd.DataFrame({"Salaray": df['salary'],
                         "Bonus": df['bonus']})
print (money_df.describe())

new = df.filter(['salary', 'bonus'], axis=1)

print (new.nlargest(5, 'salary'))
print (new.nlargest(10, 'bonus'))

df.plot.scatter('salary', 'bonus')

# Task 3: Create new feature(s)
my_dataset = data_dict
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

from sklearn import tree
# show scores of features list without new features
clf = tree.DecisionTreeClassifier()
test_classifier(clf, my_dataset, features_list)

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

# show scores of features list without new features
clf = tree.DecisionTreeClassifier()
test_classifier(clf, my_dataset, features_list)

# Store to my_dataset for easy export below.
my_dataset = data_dict
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

# Feature Selection
from sklearn.feature_selection import SelectKBest
# Univariate feature selection
# Get best features

k_best = SelectKBest(k='all')
k_best.fit(features, labels)
scores = k_best.scores_
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))

sorted_pairs_kb_df = pd.DataFrame.from_dict(sorted_pairs)
sorted_list = sorted_pairs_kb_df[0].tolist()
sorted_list.insert(0, 'poi')

# test amount of features for select k best
from amount_of_features import test_classifier_by_feature_length, calc_for_features_list_length

clf = tree.DecisionTreeClassifier()
no_features_tree = calc_for_features_list_length(clf, my_dataset, sorted_list)

no_features_tree_df = pd.DataFrame.from_dict(no_features_tree)
no_features_tree_df = no_features_tree_df.T

no_features_tree_df.plot.line(marker='.',
                              markersize=10,
                              title='Scores by number of k best features'
                              )

# Test amount of features for PCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def doPCA(number):
    pca = PCA(n_components=number)
    pca.fit(data)
    return pca


def calc_for_features_list_length_pca(my_dataset, features_list):
    count = len(features_list)
    features_scores_pca = {}
    while count > 2:
        count = count - 1
        dt = tree.DecisionTreeClassifier()

        pca = doPCA(count)

        clf = Pipeline(steps=[('pca', pca), ('dt', dt)])

        accuracy, f_score, precision, recall = test_classifier_by_feature_length(
            clf, my_dataset, features_list)
        features_scores_pca[count] = {
            'accuracy': accuracy, 'f1': f_score, 'precision': precision, 'recall': recall}

    return features_scores_pca


no_features_tree_pca = calc_for_features_list_length_pca(
    my_dataset, features_list)
no_features_tree_pca_df = pd.DataFrame.from_dict(no_features_tree_pca)
no_features_tree_pca_df = no_features_tree_pca_df.T
no_features_tree_pca_df.plot.line(marker='.',
                                  markersize=10,
                                  title='Scores by number of PCA best features'
                                  )

# store final feature list
k_best = SelectKBest(k=4)
k_best.fit(features, labels)
scores = k_best.scores_
unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))

sorted_pairs_kb_df = pd.DataFrame.from_dict(sorted_pairs)
best_features_list = sorted_pairs_kb_df[0].tolist()
best_features_list.insert(0, 'poi')
best_features_list = best_features_list[:5]
print (best_features_list)

# Task 4: Try a varity of classifiers
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
test_classifier(clf, my_dataset, best_features_list)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
svc = SVC(kernel="linear")
clf = Pipeline(steps=[('scale', StandardScaler()), ('svc', svc)])
test_classifier(clf, my_dataset, best_features_list)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
clf = Pipeline(steps=[('scale', StandardScaler()), ('knn', knn)])
test_classifier(clf, my_dataset, best_features_list)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
test_classifier(clf, my_dataset, best_features_list)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
test_classifier(rf, my_dataset, best_features_list)

# Task 5: Tune your classifier to achieve better than .3 precision and recall
from sklearn.model_selection import GridSearchCV

dt = tree.DecisionTreeClassifier()

from sklearn.model_selection import GridSearchCV

dt = tree.DecisionTreeClassifier()

pipe = Pipeline([('dt', dt)])
parameters = dict(dt__criterion=['gini', 'entropy'],
                  dt__min_samples_leaf=[1, 2, 3],
                  dt__min_samples_split=[2, 3])

clf = GridSearchCV(pipe, parameters)
clf = clf.fit(features, labels)

test_classifier(clf, my_dataset, best_features_list)

clf = GaussianNB()
dump_classifier_and_data(clf, my_dataset, best_features_list)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, best_features_list)
