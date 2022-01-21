#!conda install -c anaconda seaborn -y
#conda install -c anaconda nltk
import re
import string
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



# importing data set into dataframes with two columns: Text and Class
testData = pd.read_csv("/home/jovyan/binder/test.csv", names=["Review", "Class"], delimiter=",", header=None)
trainData = pd.read_csv("/home/jovyan/binder/train.csv", names=["Review", "Class"], delimiter=",", header=None)
valData = pd.read_csv("/home/jovyan/binder/val.csv", names=["Review", "Class"], delimiter=",", header=None)


# Count of samples in each data set
print(testData.head())
print("")
print(trainData.head())
print("")
print(valData.head())
print("")
print("Test Samples per class: {}".format(np.bincount(testData.Class)))
print("Train Samples per class: {}".format(np.bincount(trainData.Class)))
print("Val Samples per class: {}".format(np.bincount(valData.Class)))


# function used for text cleaning of input data
def clean(df):
    corpus = list()  # define empty list for corpus
    lines = df["Review"].values.tolist()  # apply text values from "Review" column to the data frame
    for text in lines: 
        text = text.lower() 
        text = re.sub(r"[,.\"!$%^&*(){}?/;`~:<>+=-]", "", text)  # regexp used to remove all special characters
        tokens = word_tokenize(text)  # splitting text
        table = str.maketrans('', '', string.punctuation) 
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        words = ' '.join(words)  # joining tokenize words together
        corpus.append(words)  # amends cleaned text to corpus
    return corpus


# applying clean function to data sets
clTest = clean(testData)
clTrain = clean(trainData)
clVal = clean(valData)


# loading TF-IDF class for feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
TF = TfidfVectorizer(min_df=15, ngram_range=(1,2)) 
xTrain = TF.fit_transform(clTrain).toarray() 
yTrain = trainData[['Class']].values
xTest = TF.transform(clTest).toarray()
yTest = testData[['Class']].values
xVal = TF.transform(clVal).toarray()
yVal = valData[['Class']].values


# loading Multinomial Naive Bayes model for text classification
from sklearn.naive_bayes import MultinomialNB
mNB = MultinomialNB()
mNB.fit(xTrain, np.ravel(yTrain)) 
y_pred_ts = mNB.predict(xTest)
y_pred_tr = mNB.predict(xTrain)
y_pred_va = mNB.predict(xVal)


# sklearn metrics used to evaluate perfomance (Accuracy) of ML model on test and val datasets and plot confusion matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
tsaccuracy = accuracy_score(yTest, y_pred_ts)
tCM = confusion_matrix(yTest, y_pred_ts)
tClasses = np.unique(yTest)

print("Test Set Accuracy:",  round(tsaccuracy,2))
print("")
print("Test Set Metrics:\n{}".format(classification_report(yTest, y_pred_ts)))
print("")

# Plot confusion matrix
fig, ax = plt.subplots()
sns.heatmap(tCM, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=tClasses,
       yticklabels=tClasses, title="Confusion matrix")
plt.yticks(rotation=0)


vlaccuracy = accuracy_score(yVal, y_pred_va)
vCM = confusion_matrix(yVal, y_pred_va)
vClasses = np.unique(yVal)

print("Val Set Accuracy:",  round(vlaccuracy,2))
print("")
print("Val Set Metrics:\n{}".format(classification_report(yVal, y_pred_va)))
print("")

# Plot confusion matrix
fig, ax = plt.subplots()
sns.heatmap(vCM, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues,
            cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=vClasses,
       yticklabels=vClasses, title="Confusion matrix")
plt.yticks(rotation=0)
