import pandas as pd
import numpy as np
import re
import random
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
# TODO: import other libraries as necessary


# Part 1a
# Load/clean/preprocess data
sentiment_train = pd.read_csv("sentiment_train.csv")

print(sentiment_train.info())
print(sentiment_train.head())

sentiment_test = pd.read_csv("sentiment_test.csv")

print(sentiment_test.info())
print(sentiment_test.head())

# Check if data has NA values
sentiment_train.shape
print('\nFeatures:', sentiment_train.columns)
train_na = sentiment_train.isna().sum()
print('\nNumber of NAs:\n')
print(train_na)
# Checked no NA values

# Not an imbalanced dataset

# Preprocess function for text
def clean_text(text):
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()
    
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    return text

# Split Sentence and target variable (Polarity) into training and testing variables
X_train = sentiment_train['Sentence']
X_test = sentiment_test['Sentence']
y_train = sentiment_train['Polarity']
y_test = sentiment_test['Polarity']
y_train = pd.to_numeric(y_train).astype(np.int64)
y_test = pd.to_numeric(y_test).astype(np.int64)


# Part 1b
# Use TFIDF Vectorizer to extract new feature from the text
tfidf_vectorizer = TfidfVectorizer(stop_words="english", preprocessor=clean_text, ngram_range=(1, 3))
# 0.78, 0.79  (ngrams 1,2)
# 0.79, 0.79  (ngrams 1,3) - best
# 0.78, 0.78  (ngrams 1,4)

# New feature generated for training and testing datasets
train_features_tdidf = tfidf_vectorizer.fit_transform(X_train)    
test_features_tdidf = tfidf_vectorizer.transform(X_test)


# Part 1c
# Train the classification model using Linear SVM Classifier (Tried Decision Tree,
# SVM, and KNeighbours, Linear SVM works the best)
model = LinearSVC(random_state=0)
model.fit(train_features_tdidf, y_train)
y_pred = model.predict(test_features_tdidf)

# Evaluate
print("\nAccuracy Score = {:.2f}".format(accuracy_score(y_test, y_pred)))
print("\nF1 Score = {:.2f}".format(f1_score(y_test, y_pred)))

# Hyperparameter tuning using grid search
tuned_parameters = [{'C': [0.001, 0.01, 0.1, 1, 10, 25, 50, 100, 1000]}]

clf = GridSearchCV(LinearSVC(), tuned_parameters, cv=5, scoring='accuracy')

clf.fit(train_features_tdidf, y_train)
print('\nBest parameter: ', clf.best_params_)
# Best hyperparameter: C: 1


# Part 1d
# Evaluate the final model with testing data
best_svm = LinearSVC(C=1, random_state=0)
best_svm.fit(train_features_tdidf, y_train)
best_svm_pred = best_svm.predict(test_features_tdidf)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, best_svm_pred))
print("\nAccuracy Score = {:.2f}".format(accuracy_score(y_test, best_svm_pred)))
print("\nF1 Score = {:.2f}".format(f1_score(y_test, best_svm_pred)))
#Accuracy: 0.79, F-1: 0.79


#Output for investigating the incorrect predictions for part 3
result = pd.DataFrame({'Sentence': X_test, 'predicted': best_svm_pred, 'Y': y_test})
result.to_csv('result.csv', index=False)