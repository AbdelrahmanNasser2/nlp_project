# -*- coding: utf-8 -*-
"""
@author: abdelrahman gamal
"""

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import re


reviews_train = []
for line in open('full_train.txt', 'r'):
    reviews_train.append(line.strip())
    
reviews_test = []
for line in open('full_test.txt', 'r'):
  reviews_test.append(line.strip())
  

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
stop_words = ['in', 'of', 'at', 'a', 'the']

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

def get_lemmatized_text(corpus):
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]


reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

reviews_train_clean = get_lemmatized_text(reviews_train_clean)
reviews_test_clean = get_lemmatized_text(reviews_test_clean)


C_values = [0.01, 0.05, 0.25, 0.5, 1]




results = []
################################################################################
# Baseline Code With Logistic Regression
print("Baseline Code With Logistic Regression")
cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
test_X = cv.transform(reviews_test_clean)


target = [1 if i < 12500 else 0 for i in range(25000)]
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.25)

# Using Logistic Regression Model
for c in C_values:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    results.append( accuracy_score(y_test, pred) )

print(results)

best_c = C_values[ np.argmax( np.array( results ) ) ] 

print(best_c)

final_model = LogisticRegression(C=best_c)
final_model.fit(X, target)
pred = final_model.predict(test_X)

print("\tPrecision: %1.3f" % precision_score(target, pred))
print("\tRecall: %1.3f" % recall_score(target, pred))
print("\tF1: %1.3f\n" % f1_score(target, pred))

print ("Final Accuracy: %s" % accuracy_score(target, pred ))
################################################################################


results = []
################################################################################
# With Bi-Grams With Logistic Regression
print("Bi-Grams With Logistic Regression")
cv = CountVectorizer(binary=True, ngram_range=(1, 2))
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
test_X = cv.transform(reviews_test_clean)


target = [1 if i < 12500 else 0 for i in range(25000)]
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.25)

# Using Logistic Regression Model
for c in C_values:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    results.append( accuracy_score(y_test, pred ) )

print(results)

best_c = C_values[ np.argmax( np.array( results ) ) ] 

print(best_c)

final_model = LogisticRegression(C=best_c)
final_model.fit(X, target)
pred = final_model.predict(test_X)

print("\tPrecision: %1.3f" % precision_score(target, pred))
print("\tRecall: %1.3f" % recall_score(target, pred))
print("\tF1: %1.3f\n" % f1_score(target, pred))

print ("Final Accuracy: %s" % accuracy_score(target, pred ))
################################################################################


results = []
################################################################################
# With Word Counts With Logistic Regression
print("Word Counts With Logistic Regression")
cv = CountVectorizer(binary=False)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
test_X = cv.transform(reviews_test_clean)


target = [1 if i < 12500 else 0 for i in range(25000)]
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.25)

# Using Logistic Regression Model
for c in C_values:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    results.append( accuracy_score(y_test, pred ) )

print(results)

best_c = C_values[ np.argmax( np.array( results ) ) ] 

print(best_c)

final_model = LogisticRegression(C=best_c)
final_model.fit(X, target)
pred = final_model.predict(test_X)

print("\tPrecision: %1.3f" % precision_score(target, pred))
print("\tRecall: %1.3f" % recall_score(target, pred))
print("\tF1: %1.3f\n" % f1_score(target, pred))

print ("Final Accuracy: %s" % accuracy_score(target, pred ))
################################################################################

results = []
################################################################################
# With TF-IDF With Logistic Regression
print("TF-IDF With Logistic Regression")
cv = TfidfVectorizer()
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
test_X = cv.transform(reviews_test_clean)


target = [1 if i < 12500 else 0 for i in range(25000)]
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.25)

# Using Logistic Regression Model
for c in C_values:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    results.append( accuracy_score(y_test, pred ) )

print(results)

best_c = C_values[ np.argmax( np.array( results ) ) ] 

print(best_c)

final_model = LogisticRegression(C=best_c)
final_model.fit(X, target)
pred = final_model.predict(test_X)

print("\tPrecision: %1.3f" % precision_score(target, pred))
print("\tRecall: %1.3f" % recall_score(target, pred))
print("\tF1: %1.3f\n" % f1_score(target, pred))

print ("Final Accuracy: %s" % accuracy_score(target, pred ))
################################################################################

results = []
################################################################################
# Using Bi-Gram With Support Vector Machine Model  
print("Bi-Gram With Support Vector Machine Model")
cv = CountVectorizer(binary=True, ngram_range=(1, 2))
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
test_X = cv.transform(reviews_test_clean)


target = [1 if i < 12500 else 0 for i in range(25000)]
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.25)

# Using SVM Model
for c in C_values:
    svm = LinearSVC(C=c)
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    results.append( accuracy_score(y_test, pred ) )
    
print(results)

best_c = C_values[ np.argmax( np.array( results ) ) ] 

print(best_c)

# Using Support Vector Machine Model    
final_svm_ngram = LinearSVC(C=best_c)
final_svm_ngram.fit(X, target)
pred = final_svm_ngram.predict(test_X)

print("\tPrecision: %1.3f" % precision_score(target, pred))
print("\tRecall: %1.3f" % recall_score(target, pred))
print("\tF1: %1.3f\n" % f1_score(target, pred))

print ("Final Accuracy: %s" % accuracy_score(target, pred ))
################################################################################
