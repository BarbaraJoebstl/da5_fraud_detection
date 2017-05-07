import pickle
import sys
import numpy
import pandas as pd

from sklearn.cross_validation import StratifiedShuffleSplit
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


def test_classifier_by_feature_length(clf, dataset, feature_list, folds=1000):
    data = featureFormat(dataset, feature_list, sort_keys=True)

    labels, features = targetFeatureSplit(data)

    cv = StratifiedShuffleSplit(labels, folds, random_state=42)

    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        # fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + \
            false_negatives + false_positives + true_positives

        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / \
            (2 * true_positives + false_positives + false_negatives)

    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

    return accuracy, f1, precision, recall


def calc_for_features_list_length(clf, dataset, best_features_list):
    '''
    pass the feature list, ranked by best features and see how the scores
    work depending on the number of features
    '''

    count = len(best_features_list)
    features_scores = {}

    while count > 2:
        best_features_list.pop()
        count = count - 1
        accuracy, f_score, precision, recall = test_classifier_by_feature_length(
            clf, dataset, best_features_list)
        features_scores[count] = {
            'accuracy': accuracy, 'f1': f_score, 'precision': precision, 'recall': recall}

    return features_scores
