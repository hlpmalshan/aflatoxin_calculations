import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
import random

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib.colorbar import ColorbarBase
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

numOfImgs = 13
numberOfSamples =15
numOfLevels = 8
pixelSize = 10

def plot_confusion_matrix(clf, X, y_true, ax=None, normalize=False,  true_labels=None, pred_labels=None):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if ax is None:
        ax = plt.gca()
    
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', ax=ax, annot_kws={"fontsize": 12, "fontfamily": "Times New Roman"})

    if true_labels:
        ax.set_yticklabels(true_labels, fontsize = 12, fontname='Times New Roman')
    if pred_labels:
        ax.set_xticklabels(pred_labels, fontsize = 12, fontname='Times New Roman')

    ax.set_xlabel('Predicted Contamination Level (μg/kg)', fontsize = 12, fontname='Times New Roman')
    ax.set_ylabel('True Contamination Level (μg/kg)', fontsize = 12, fontname='Times New Roman')

# Load the dataset from the CSV file
dataset = pd.read_csv('Superpixel dataset.csv', header = None)

# Convert the new_dataset to a DataFrame for easier manipulation
df = pd.DataFrame(dataset)

# Separate the features and labels
features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

# Create empty lists to store the train and test data
train_data = []
test_data = []

# Define the number of chunks to select from each class
num_chunks = 1  # Change this to select more or fewer chunks

'''
[0, 2, 5, 10, 15, 20, 30, 40]
[0, 1, 2,  3,  4,  5,  6,  7,]
'''

for i in [ 0, 1, 2, 3, 4, 5, 6, 7]: #5 
    # Get all the rows of this class
    class_i_data = df[df.iloc[:, -1] == i]
    
    # Split the class data into chunks of 100 rows
    class_i_chunks = np.array_split(class_i_data, len(class_i_data) // 100)
    
    # Randomly select num_chunks for testing
    test_chunks = random.sample(class_i_chunks, num_chunks)
    
    # Add the selected chunks to the test data
    test_data.extend(test_chunks)
    
    # Add the remaining chunks to the train data
    train_data.extend(chunk for chunk in class_i_chunks if not any(chunk.equals(test_chunk) for test_chunk in test_chunks))

# Convert the lists back to DataFrames
train_data = pd.concat(train_data)
test_data = pd.concat(test_data)

# Convert the DataFrames back to numpy arrays, if necessary
train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

print('Train ', train_data.shape)
print('Test ', test_data.shape)

train_features = train_data[:, :-1]
train_labels = train_data[:, -1]

test_features = test_data[:, :-1]
test_labels = test_data[:, -1]

# Create an instance of LDA
lda = LDA()

# Fit LDA to the dataset
lda.fit(train_features, train_labels)

# Transform the dataset to the LDA space
lda_features = lda.transform(train_features)
lda_testData = lda.transform(test_features)

#The following code constructs the Scree plot
per_var = np.round(lda.explained_variance_ratio_* 100, decimals=1)
print('LDA Explained Variance - ', per_var)

X_train = lda_features
y_train = train_labels

X_test = lda_testData
y_test = test_labels

# STACKING
estimators = [ ('gnb', GaussianNB()), ('log_reg', LogisticRegression()), ('svc', SVC()), ('dt', DecisionTreeClassifier()), ('knn', KNeighborsClassifier()) ]

# GRIDSEARCH
# define the model with default hyperparameters
model = StackingClassifier(estimators = estimators)

# define the grid of values to search
grid = dict()
grid['final_estimator'] = [SVC()]
grid['passthrough'] = [True, False]

# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# execute the grid search
grid_result = grid_search.fit(X_train, y_train)

# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Get the best parameters found by grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Make predictions on the test set using the best model
best_adaboost_clf = grid_search.best_estimator_
y_pred = best_adaboost_clf.predict(X_test)

# Evaluate the accuracy of the best model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Define custom labels for true and predicted labels
custom_true_labels = ['0.00', '9.30', '23.26', '46.52', '69.78', '93.04', '139.56', '186.08']
custom_pred_labels = ['0.00', '9.30', '23.26', '46.52', '69.78', '93.04', '139.56', '186.08']

# Plot confusion matrix for test data
plt.figure(figsize=(8, 6))
plot_confusion_matrix(best_adaboost_clf, X_test, y_test, normalize=True, true_labels=custom_true_labels, pred_labels=custom_pred_labels)
plt.savefig('Confusion Matrix.png', dpi = 1000)
plt.show()
