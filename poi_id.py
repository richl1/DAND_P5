#!/usr/bin/python
import pandas as pd
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, load_classifier_and_data, test_classifier
from sklearn.cross_validation import train_test_split

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from itertools import compress
from pprint import pprint
from IPython.display import display
from time import time

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Convert to Pandas
df = pd.DataFrame.from_records(list(data_dict.values()))
employees = pd.Series(list(data_dict.keys()))
# set the index of df to be the employees series:
df.set_index(employees, inplace=True)
# Convert Numeric Column types
# Convert Financial Columns 
df[['bonus','deferral_payments',
    'deferred_income', 'director_fees','exercised_stock_options','expenses',
    'loan_advances', 'long_term_incentive', 'loan_advances','other', 
    'restricted_stock', 'restricted_stock_deferred','salary','total_payments', 
    'total_stock_value']] = df[['bonus','deferral_payments',
    'deferred_income', 'director_fees','exercised_stock_options','expenses',
    'loan_advances', 'long_term_incentive', 'loan_advances','other', 
    'restricted_stock', 'restricted_stock_deferred','salary','total_payments', 
    'total_stock_value']].apply(pd.to_numeric,errors='coerce')
# Convert email columns
df[['from_messages', 'from_poi_to_this_person','from_this_person_to_poi',
    'shared_receipt_with_poi','to_messages']] = df[['from_messages', 
    'from_poi_to_this_person','from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages']].apply(pd.to_numeric,errors='coerce')

### Task 2: Remove outliers
df.drop('TOTAL', inplace = True)    # Remove Outliers
df.drop('THE TRAVEL AGENCY IN THE PARK', inplace = True)    # Remove Outliers
print "Removed Outliers for 'TOTAL' and 'THE TRAVEL AGENCY IN THE PARK'\n"


### Task 3: Create new feature(s)
df['from_poi_to_this_person_pct'] = \
    df['from_poi_to_this_person'] / df['to_messages']
df['from_this_person_to_poi_pct'] = \
    df['from_this_person_to_poi'] / df['from_messages']

nan_observations = {}
for column in df:
    nan_observations[column] = df[column].isnull().sum()

df.fillna(value=0,inplace = True)

### Data Exploration
print "Total number of data points (observations) :", len(df.index)
print "Numer of POI observations :", len(df['poi'][df['poi']])
print "Number of non-POI obseervations :", len(df['poi'][df['poi'] == False])


### Store to my_dataset for easy export below.
df.to_csv('enron_for_eda.txt')
df_dict = df.to_dict('index')
my_dataset = df_dict

features_list = ['poi',
 'bonus',
 'deferral_payments',
 'deferred_income',
 'director_fees',
 'exercised_stock_options',
 'expenses',
 'from_messages',
 'from_poi_to_this_person',
 'from_this_person_to_poi',
 'loan_advances',
 'long_term_incentive',
 'other',
 'restricted_stock',
 'restricted_stock_deferred',
 'salary',
 'shared_receipt_with_poi',
 'to_messages',
 'total_payments',
 'total_stock_value',
 'from_poi_to_this_person_pct',
 'from_this_person_to_poi_pct'
]

# create a list of features without 'poi' which is a label
features_no_poi = list(features_list) # copy the feature list
features_no_poi.pop(0)  # remove 'poi' from the list
print "\nTotal features available : ", len(features_no_poi)
print
print "Available Feature sorted by NaNs"
df_nans = pd.DataFrame.from_dict(nan_observations, orient = 'index')
pprint(df_nans.sort_values(by = 0, ascending=False))


####################################################################
# Implement Modeling Pipeline with "Select K Best" and "Naive Bayes"
####################################################################
print "\n******************\n Select K Best + Gaussian NB Pipeline\n"
t0 = time()
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

k_range = range(2,10)
params = {'SKB__k' : k_range }
pipeline = Pipeline([('SKB', SelectKBest()), ('classifier', GaussianNB())])
cv = StratifiedShuffleSplit(labels, 100, test_size=0.2, random_state=60)
gs = GridSearchCV(pipeline, params, cv=cv, scoring="f1_weighted")
gs.fit(features, labels)
clf = gs.best_estimator_

# Print the selected features and pvalues
print "Processing time:", round(time()-t0, 3), "s"
k_best_support = clf.named_steps['SKB'].get_support(False).tolist()
df_selected_features1 = pd.DataFrame(
    {'Feature': list(compress(features_no_poi, k_best_support)),
     'p value': list(compress(clf.named_steps['SKB'].pvalues_,k_best_support))
    })
pprint(df_selected_features1)
print

# Test the results
dump_classifier_and_data(clf, my_dataset, features_list)
clf, dataset, feature_list = load_classifier_and_data()
test_classifier(clf, dataset, feature_list)

#####################################################################
# Implement Modeling Pipeline with "Select K Best" and "DecisionTree"
####################################################################
print "\n******************\n Select K Best + DecisionTree Pipeline\n"
t0 = time()
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

k_range = range(2,8)
params = {'SKB__k' : k_range,
          "dt__min_samples_leaf": [2, 4, 6],
          "dt__min_samples_split": [8, 10, 12],
          "dt__min_weight_fraction_leaf": [0, 0.1],
          "dt__criterion": ["gini", "entropy"],
          "dt__random_state": [42, 46]}
          
pipeline = Pipeline([('SKB', SelectKBest()),('dt', DecisionTreeClassifier())])
cv = StratifiedShuffleSplit(labels, 100, test_size=0.2, random_state=60)
gs = GridSearchCV(pipeline, params, cv=cv, scoring="f1_weighted")
gs.fit(features, labels)
clf = gs.best_estimator_

# Print the selected features, pvalues, and DT Importances
print "Processing time:", round(time()-t0, 3), "s"
k_best_support = clf.named_steps['SKB'].get_support(False).tolist()
df_selected_features2 = pd.DataFrame(
    {'Feature': list(compress(features_no_poi, k_best_support)),
    'p value': list(compress(clf.named_steps['SKB'].pvalues_,k_best_support)),
    'Importance' : clf.named_steps['dt'].feature_importances_.tolist()
    })
pprint(df_selected_features2)
print

# Test the results
dump_classifier_and_data(clf, my_dataset, features_list)
clf, dataset, feature_list = load_classifier_and_data()
test_classifier(clf, dataset, feature_list)

#####################################################################
# Implement "Naive Bayes" with Manually Selected Features
#####################################################################
print "\n******************\n Gaussian NB w/ manual Features\n"
features_list_manual = ['poi',
 'from_poi_to_this_person_pct',
 'salary',  'deferred_income', 
 'exercised_stock_options',  'expenses', 
 'total_stock_value']

print "Manually Selected Features : ", features_list_manual[1:]

t0 = time()
data = featureFormat(my_dataset, features_list_manual, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test\
    = train_test_split(features, labels, test_size=0.2, random_state=42)
clf = GaussianNB()
print "Processing time:", round(time()-t0, 3), "s"

# Test the results
dump_classifier_and_data(clf, my_dataset, features_list_manual)
clf, dataset, features_list_manual = load_classifier_and_data()
test_classifier(clf, dataset, features_list_manual)