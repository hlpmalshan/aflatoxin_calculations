import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math

from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, exp, Mul , pi, sqrt, DiracDelta
from sympy.utilities.lambdify import lambdify
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib.colorbar import ColorbarBase

# Set the font to Times New Roman for both normal text and math text
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'

reference_level = 0
numberOfSamples = 15
numberOfLevels = 8
numberOfvalidationSamples = 3
numberOfVaidationLevels = 7

def separate_data_by_level(df, level): #Used the Dataframe
    levelData = df[df['Labels'] == level]
    return levelData


def separate_data_by_sample(df):
    for i in range (numberOfSamples):
        return dataset.iloc[i*100:i*100+100,:]

# Load the dataset from the CSV file
dataset = pd.read_csv('Superpixel dataset.csv')
validation = pd.read_csv('Validation Superpixel dataset.csv')

# Extract the feature vectors and labels
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

valfeatures = validation.iloc[:, :-1].values
validationlabels = validation.iloc[:, -1].values

# Create an instance of LDA
lda = LDA()

# Fit LDA to the dataset
lda.fit(features, labels)

# Transform the dataset to the LDA space
lda_features = lda.transform(features)

# Project the validation dataset onto the same LDA axes obtained from the training dataset
validation_lda_features = lda.transform(valfeatures)

dataset = np.hstack((lda_features, labels))
validationdataset = np.hstack((validation_lda_features, validationlabels))

def get_meanVector(df):   #Used Dataframe
    meanVector = df.groupby('Labels').mean().values.T
    return meanVector

def get_CovarianceMatrix(refData,meanVector): #used Numpy
    dfNP = refData.iloc[:, :-1].values
    dfNP = dfNP - meanVector.T
    covarianceMatrix = np.dot(dfNP.T, dfNP)
    return covarianceMatrix


def get_bhattacharyyaDistance(levelData):
    level_meanVector = get_meanVector(levelData)
    level_CovMatrix = get_CovarianceMatrix(levelData,level_meanVector)
    mean_diff = ref_meanVector - level_meanVector
    mean_cov = 0.5*(ref_CovMatrix + level_CovMatrix)
    det_mean_cov = np.linalg.det(mean_cov)
    ref_Cov_Det = np.linalg.det(ref_CovMatrix)
    level_Cov_Det = np.linalg.det(level_CovMatrix)
    
    BhattDistance = (1/8)*(mean_diff.T).dot(np.linalg.inv(mean_cov)).dot(mean_diff) + (0.5)*(np.log(det_mean_cov/np.sqrt(abs(ref_Cov_Det)*abs(level_Cov_Det))))
    return np.sum(BhattDistance)

refData = separate_data_by_level(dataset, 0)
refData = refData[:100]
ref_meanVector = get_meanVector(refData)
ref_CovMatrix = get_CovarianceMatrix(refData,ref_meanVector)
BhattachryyaDisMat = np.empty((numberOfSamples,numberOfLevels))

# Initialize the Bhattacharyya and JM distance matrices for the validation dataset
BhattachryyaDisMat_validation = np.empty((numberOfvalidationSamples, numberOfVaidationLevels))

# Define the sequence of levels
selected_level = 6  # Change this to the desired level
levels = [0, 1, 2, 3, 4, 5, 6, 7]
# Loop over each level in the sequence
for i, level in enumerate(levels):
    # Round the level to the nearest integer for indexing
    level = round(level)
    levelData = separate_data_by_level(dataset, level)
    
    if level == selected_level:
        for j in range(numberOfSamples):
            sampleDataSet = levelData.iloc[j * 100:j * 100 + 100, :]
            BhattDistance = get_bhattacharyyaDistance(sampleDataSet)
            
    for j in range(numberOfSamples):
        sampleDataSet = levelData.iloc[j*100:j*100+100,:]
        BhattDistance = get_bhattacharyyaDistance(sampleDataSet)
        BhattachryyaDisMat[j][i] = BhattDistance
        
for i,level in enumerate(range (0,numberOfLevels*1,1)):
    levelData = separate_data_by_level(dataset, level)

    if level == selected_level:
        for j in range(numberOfSamples):
            sampleDataSet = levelData.iloc[j * 100:j * 100 + 100, :]
            BhattDistance = get_bhattacharyyaDistance(sampleDataSet)

selected_validation_level = 28  # Change this to the desired level
vali = [4,8,10,12,16,24,28]
# Loop over each level and sample in the validation dataset
for i, level in enumerate(vali):
    levelData = separate_data_by_level(validationdataset, level)
    
    if level == selected_validation_level:
        for j in range(numberOfvalidationSamples):
            sampleDataSet = levelData.iloc[j * 100:j * 100 + 100, :]
            BhattDistance = get_bhattacharyyaDistance(sampleDataSet)
            
    for j in range(numberOfvalidationSamples):
        sampleDataSet = levelData.iloc[j*100:j*100+100,:]
        BhattDistance = get_bhattacharyyaDistance(sampleDataSet)
        BhattachryyaDisMat_validation[j][i] = BhattDistance

##########################################################################################
# Reshape the matrix into a 1D array
#flattened_data = JMDisMat.ravel()
flattened_data = BhattachryyaDisMat.ravel()
flattened_data_validation = BhattachryyaDisMat_validation.ravel()

# Create x-values corresponding to columns of the matrix, repeated for each row
# Replace the existing percentages with your desired values
percentages = [0, 9.30, 23.26, 46.52, 69.78, 93.04, 139.56, 186.08]
percentages_validtion = [18.61, 37.22, 46.52, 55.82, 74.43, 111.65, 130.26]

x_values = np.concatenate((np.tile(percentages, 15), np.tile(percentages_validtion, numberOfvalidationSamples)))

# Plot the scatter plot for training data in blue color
plt.figure(1,figsize=(10, 6))
scatter = plt.scatter(x_values[:len(flattened_data)], flattened_data, color='blue') #training data

# Add labels and title
plt.xlabel("Aflatoxin contamination level (µg/kg)",fontsize=18)
plt.ylabel("Bhattacharyya distance",fontsize=19)
plt.grid(True, linestyle='--')
plt.tick_params(axis='both', labelsize=16)

# Perform polynomial regression of degree 1 (you can change the degree as needed)
degree = 1
coefficients = np.polyfit(x_values, np.concatenate((flattened_data, flattened_data_validation)), degree)

# Generate the curve using the polynomial coefficients
x_fit = np.linspace(x_values.min(), x_values.max(), 100)
y_fit = np.polyval(coefficients, x_fit)

# Plot the fitted curve
line,= plt.plot(x_fit, y_fit, color='red', linewidth = 2.3, label=f'Fitted Curve: y = {" + ".join([f"{coeff:.4f} * p^{degree - i}" for i, coeff in enumerate(coefficients)])}')

# Calculate the R-squared value
y_mean = np.mean(np.concatenate((flattened_data, flattened_data_validation)))
y_pred = np.polyval(coefficients, x_values)
ss_tot = np.sum((np.concatenate((flattened_data, flattened_data_validation)) - y_mean) ** 2)
ss_res = np.sum((np.concatenate((flattened_data, flattened_data_validation)) - y_pred) ** 2)
r_squared = 1 - (ss_res / ss_tot)
r_squared = round(r_squared, 4)

# Calculate Root Mean Square Error (RMSE)
rmse = np.sqrt(np.mean((np.concatenate((flattened_data, flattened_data_validation)) - y_pred) ** 2))

# Calculate RSS
rss = np.sum((np.concatenate((flattened_data, flattened_data_validation)) - y_pred) ** 2)

# Define a function to calculate AIC
def calculate_aic(n, k, rss):
    return n * np.log(rss / n) + 2 * k

# Calculate AIC value
n = len(np.concatenate((flattened_data, flattened_data_validation)))
k = degree + 1  # number of parameters (degree + intercept)
aic = calculate_aic(n, k, rss)

# Output the fitted polynomial equation and R-squared value
equation = ' + '.join([f'{coeff:.4f} * p^{degree - i}' if i == 0 else f'{coeff:.4f} * p^{degree - i}' for i, coeff in enumerate(coefficients)])
print(equation)
print("R^2 Value:", r_squared)
print("RMSE:", rmse)
print("AIC:", aic)

# Create the legend labels
poly_equation = ' + '.join([f'{coeff:.4f}' if i == degree else f'{coeff:.4f}p' if i == degree - 1 else f'{coeff:.4f}p^{degree - i}' for i, coeff in enumerate(coefficients)])
calibrated_curve_label = f'Fitted Curve: y = {poly_equation} : $R^2$ = {r_squared}'
samples_label = 'Samples (Calibration)'
# Add the legend to the plot
# Add the legend to the plot with correct order and labels
plt.legend([line, scatter], [calibrated_curve_label,samples_label],edgecolor='black',fontsize=16)
# Show the plot
plt.rcParams['text.usetex'] = False
#Save the figure with a high resolution
plt.savefig('Bhattachryya Distance vs AFB1 Calibration graph.png', dpi=1000)
plt.show()


# Calculate the predicted values for the validation data
y_pred_validation = np.polyval(coefficients, x_values[len(flattened_data):])

# Calculate the R-squared value for the validation data
y_mean_validation = np.mean(flattened_data_validation)
ss_tot_validation = np.sum((flattened_data_validation - y_mean_validation) ** 2)
ss_res_validation = np.sum((flattened_data_validation - y_pred_validation) ** 2)
r_squared_validation = 1 - (ss_res_validation / ss_tot_validation)
r_squared_validation = round(r_squared_validation, 4)

# Calculate Root Mean Square Error (RMSE) for the validation data
rmse_validation = np.sqrt(np.mean((flattened_data_validation - y_pred_validation) ** 2))

# Print the R^2 value and RMSE for the validation data
print("Validation R^2 Value:", r_squared_validation)
print("Validation RMSE:", rmse_validation)

# Get the equation of the fitted curve
poly_equation = ' + '.join([f'{coeff:.4f}' if i == degree else f'{coeff:.4f}p' if i == degree - 1 else f'{coeff:.4f} p^{degree - i}' for i, coeff in enumerate(coefficients)])

# Create the legend labels
calibrated_curve_label = f'Calibrated Curve: B = {poly_equation}'
validation_data_label = f'Validation Data : $R^2$ = {r_squared_validation}'


plt.rc('xtick', labelsize=16)  # Increase x-axis tick label font size
plt.rc('ytick', labelsize=16)  # Increase y-axis tick label font size
# Second figure: only the fitted polynomial and validation points
plt.figure(2,figsize=(10, 6))
plt.plot(x_fit, y_fit, color='black', linewidth=2.3, label=calibrated_curve_label)  # fitted polynomial
plt.scatter(x_values[len(flattened_data):], flattened_data_validation, color='red', label=validation_data_label)  # validation data
plt.xlabel('Aflatoxin contamination level (µg/kg)',fontsize=18)
plt.ylabel('Bhattacharyya distance',fontsize=19)
plt.legend(edgecolor='black',fontsize=16)
plt.grid(True, linestyle='--')
plt.xlim(left=0)
plt.ylim(bottom=0)
# Save the figure with a high resolution
plt.savefig('Bhattachryya Distance vs AFB1 Validation graph.png', dpi=1000)
plt.show()
