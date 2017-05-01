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

In the dataset there are 18 persons flagged as POI and 128 as non POI.

#### Outlier Detection
*Outliers*.There is one datapoint far far away from the others. As the name of this datapoint is "Total " it is clear that this is an outlier due to a spreadsheet error.  This datapoint will be removed.

We can identify some outliers in the salary. SKILLING JEFFREY K, LAY KENNETH L and FREVERT MARK A had a salary over 1.000.000 each. The highest Bonus had LAVORATO JOHN J. There are some more outliers, but we want to keep this data for our analysis, because those are more likely to be anomalies than outliers.

*Missing Data*.
- Feauteres: When we take a closer look at the feautures, we can see that 
`restricted_stock_deferred`, `director_fees` and `loan_advances` have at least 88.7% of NaN values. We remove them from our features list.

- Datapoints:
There are four entries that have 13 or more NaN s out of 15. We remove "LOCKHART EUGENE E" because this entry only contains NaNs and we will also remove "THE TRAVEL AGENCY IN THE PARK", because it contains only on entry and is not a person.

### Optimize Feature Selection
After removing the features with the most NaN values, we engineer new features to get a better understanding of the interaction between POIs and non POIs.
The new features are:
-- `fraction_to_poi`. from_this_person_to_poi/to_messages
-- `fraction_from_poi`. from_poi_to_this_person/from_messages
-- `fraction_bonus`. total_payments/bonus

For the feature selection I ended up using PCA, because it takes the features and creates new ones superior to original attributes.

Principal Component of a dataset is the direction that has the largest variance*, because it retains the max amoumt of information of the original data.

### Algorithm
I tried several classifiers _with_ PCA.

| Classifier | Accuracy | Precision  | Recall  | F1  |
| ------------- |:--------:|:--------:|:--------:|
| GaussianNB | 0.85653 | 0.45498 | 0.38400 | 0.41649 |
| SVC | 0.85807 | 0.36864 | 0.09050 | 0.14532 |
| Knn | 0.86080 | 0.13934 |0.00850 | 0.01602 |
| DecisionTree | 0.79567 | 0.24631 | 0.25850 | 0.25226 |
| RandomForest | 0.85240 | 0.36352 | 0.14250 | 0.20474 |

The GaussianNB peformed the best, with the highest values for all evaluation metrics.

With the help of GridSearchCV, wich is a way of systematically working through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. We are able to tune the DecisionTree classifier.

| Classifier with GridSearchCV | Accuracy | Precision  | Recall  | F1  |
| ------------- |:--------:|:--------:|:--------:|
| DecisionTree | 0.85653 | 0.45498 | 0.38400 | 0.41649 |

### Validate and Evaluate
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

## Run
In the _final_project_ folder you can:
 - run a python notebook named `da5_fraud_detection.ipynb`
 - or you can simply run `poi_id.py`

## Data
 The data and the starter code that reads in the data, takes my features of choice, then puts them into a numpy array is provided by [Udacity](https://github.com/udacity/ud120-projects).
