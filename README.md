# Fraud Detection
 This assignment belongs to the "Intro into Machine Learning" Course, which is part of the of the Udacity Nanodegree Data Analyst.

## Project Overview
 In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.

## Goal
The goal of this project is to create a predictive model for identifying "Persons of interest (POI)" with the help of machine learning methods. Those people are said to take part in financial fraud.
 
### Data Exploration
#### Dataset
The dataset consists of 146 datapoints, where each datapoint represents a person. The features given are those:

*financial features*. ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] (all units are in US dollars)

*email features*. ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

*POI label*.[‘poi’] target feature(boolean, represented as integer)

In the dataset there are 18 persons flagged as POI and 128 as non POI. This shows a clear class imbalance. Most machine learning algorithms work the best when the numbers are roughly equal.

#### Outlier Detection
*Outliers*.There is one datapoint far far away from the others. As the name of this datapoint is "Total " it is clear that this is an outlier due to a spreadsheet error.  This datapoint will be removed.

We can identify some outliers in the salary. SKILLING JEFFREY K, LAY KENNETH L and FREVERT MARK A had a salary over 1.000.000 each. The highest Bonus had LAVORATO JOHN J. There are some more outliers, but we want to keep this data for our analysis, because those are more likely to be anomalies than outliers.

*Missing Data*.
- Feauteres: When we take a closer look at the feautures, we can see that 
`restricted_stock_deferred`, `director_fees` and `loan_advances` have at least 88.7% of NaN values. But for now we will not remove them, because even sparse data can predict well.

- Datapoints:
There are four entries that have 13 or more NaN s out of 15. We remove "LOCKHART EUGENE E" because this entry only contains NaNs and we will also remove "THE TRAVEL AGENCY IN THE PARK", because it contains only on entry and is not a person.

### Optimize Feature Selection

After removing the features with the most NaN values, we engineer new features to get a better understanding of the interaction between POIs and non POIs.

We want to create new features showing the fraction of the POI messages sent, because we think that the messaging between POIs relatively higher than between non POIs. We also suggest that the bonus for POIs is relatively higher than for non POIs.

The new features are:
-- `fraction_to_poi`. from_this_person_to_poi/to_messages
-- `fraction_from_poi`. from_poi_to_this_person/from_messages
-- `fraction_bonus`. total_payments/bonus

With the new features we are able to improve the accuracy from 0.79015 to 0.79146 and also the other scores got a little bit better.

## Amount of Features

    By selecting (the best) features we want to 
        - reduce overfitting
        - improve accuracy
        - reduce training time

    Because we were not sure how many features and which feature selection to use, we plotted two charts to compare SelectKBest and PCA. We plotted the scores depending on the number of best features.

    As the plot above shows the best scores can be achieved with using the best 19 features. But the best scores of PCA are worse than those of SelectK Best. Regarding the calculating time of the program and the scores, we suggest that is is better to use SelectK Best, because only 4 features are needed to get scores around 0.4.



### Algorithm

#### Feature Scaling

- *NaiveBayes.* Does Feature scaling by design, so we don't need to scale our features before.
- *SVM.* Needs Feature Scaling, because SVMs assume that the data it works with is in a standard range, usually either 0 to 1, or -1 to 1.
- *Knn.* Because this algorithm measure the distances between pairs of samples, we have to scale the features.
- *DecisionTree, RandomForest.* Do not require feature scaling.

We tried several classifiers _with_ PCA and n_components = 10:

| Classifier | Accuracy | Precision  | Recall  | F1  |
| --- | :--------: | :--------: | :--------: | :--------: |
| GaussianNB | 0.85653 | 0.45498 | 0.38400 | 0.41649 |
| SVC | 0.85807 | 0.36864 | 0.09050 | 0.14532 |
| Knn | 0.86080 | 0.13934 |0.00850 | 0.01602 |
| DecisionTree | 0.79567 | 0.24631 | 0.25850 | 0.25226 |
| RandomForest | 0.85240 | 0.36352 | 0.14250 | 0.20474 |

After optimizing the feature selection with SelectKBest and k=4, we were able to achieve these scores:

| Classifier | Accuracy | Precision  | Recall  | F1  |
| --- | :--------: | :--------: | :--------: | :--------: |
| GaussianNB | 0.84677 | 0.50312 | 0.32300 | 0.39342 |
| SVC | 0.84677 | 0.53351 | 0.09950 | 0.16772 |
| Knn | 0.84185 | 0.37273 | 0.04100 | 0.07387 |
| DecisionTree | 0.78792 | 0.31847 | 0.33200 | 0.32509 |
| RandomForest | 0.84485 | 0.49077 | 0.22600 | 0.30948 |

Altough the boost of the accuracy sank for every classifier at about 0.01, we were able to increase precision, recall and f1

The GaussianNB peformed the best, with the highest values for all evaluation metrics.

With the help of GridSearchCV, wich is a way of systematically working through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. We are able to boost the accuracy and the precision of DecisionTree classifier, but the recall got worse. 
So we will use the GaussianNB, because we achieved the best result in the shortest time.

### Validation

As seen above we have used some validation metrics.
Validation is important because it gives an estimate of the performance of an independent dataset and serves as a check for overfitting.

- *Recall*.
true positives / true positives + false negatives
_good recall_. Whenever the target (in our case POI) shows up in the data set, we can identify it almost every time. he cost of this is that we sometimes get some false positives.
 
- *Precision*. 
true positives / true positives + false positives
_good precision_. Whenever the target (in our case POI) gets flagged in the data set, it is very likely to be a real target and not a false alarm.  

- *F1 Score*.
This score is a measure of a test's accuracy. It considers recall and precision.
_good f1 score_. This is the best of both worlds. Both my false positive and false negative rates are low.

### Evaluation

Validation is process of determining how well your model performs. We are using k-fold cross validation for this project. This means the data ist split into test (best validation, when maximum amount) and training (best learning results, when maximum amount). In order to handle the trade off between the split data, we run k separate learning experiment, so at the end all data has been used for training and testing.

In our case the  sklearn StratifiedShuffleSplit with labels, folds=1000 and random_state = 42 as parameters is used as a cross-validator in the tester.py

## Run
In the _final_project_ folder you can:
 - run a python notebook named `da5_fraud_detection.ipynb`
 - or you can simply run `poi_id.py`

## Data
 The data and the starter code that reads in the data, takes my features of choice, then puts them into a numpy array is provided by [Udacity](https://github.com/udacity/ud120-projects).
