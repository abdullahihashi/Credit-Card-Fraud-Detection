# Importing necessary Packages and libraries
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__))


# Loading the necessary dataset from the file
data = pd.read_csv('creditcardfrauddetection.csv')


# Determining the number of fraudulent cases in the dataset
fraudulent = data[data['Class'] == 1]
validTransactions = data[data['Class'] == 0]

outlier_fraction = len(fraudulent)/float(len(Valid))
print(outlier_fraction)

print('fraudulent Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))


# -------- Local Outlier Factor and Isolation Forest Algorithm ----------


# set random state
state = 1

# Defining outlier detection
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X), contamination=outlier_fraction, random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)}


# Fit the model with dataset
plt.figure(figsize=(9, 7))
number_outliers = len(Fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):

    # Fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_prediction = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_prediction = clf.predict(X)

    # Set the prediction values to 1 for fraudulent and 0 for valid tranaction.
    y_prediction[y_prediction == 1] = 0
    y_prediction[y_prediction == -1] = 1

    number_errors = (y_prediction != Y).sum()

    # Run classification metrics
    print('{}: {}'.format(clf_name, number_errors))
    print(accuracy_score(Y, y_prediction))
    print(classification_report(Y, y_prediction))
