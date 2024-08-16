import os
import cv2
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter

numOfImgs = 14
numberOfSamples = 15
numOfLevels = 8
pixelSize = 10


def wiener_filter(image, w):
    # Initialize padded_image
    pad_amount = w // 2
    padded_image = np.pad(image, pad_width=((pad_amount, pad_amount), (pad_amount, pad_amount)), mode='reflect')

    # Compute local mean and variance
    mean = uniform_filter(padded_image, size=(w, w))
    mean_sq = uniform_filter(padded_image**2, size=(w, w))
    variance = mean_sq - mean**2

    # Avoid division by zero
    #variance[variance == 0] = 1e-10

    # Compute white noise
    white_noise = np.sum(variance[pad_amount:pad_amount+image.shape[0], pad_amount:pad_amount+image.shape[1]]) / (image.shape[0] * image.shape[1])

    # Apply Wiener filter
    filtered_image = mean + (padded_image - mean) * variance / (variance + white_noise)

    return filtered_image[pad_amount:pad_amount+image.shape[0], pad_amount:pad_amount+image.shape[1]]


def MakeMainMatrix_DarkCurrentImage():
    pwd_main = os.getcwd()
    folders = glob(pwd_main + "/*/", recursive = True)
    folders = sorted(folders)
    #print(folders)
    
    mainMatrix = np.empty((0,13))
    for f in folders:
        subFolders = glob(f + "/*/", recursive = True)
        subFolders = sorted(subFolders)
        
        imageMatrixforOneLevel = np.ones((0,13))
        avgMatrix = np.ones((1,numOfImgs))
        
        for sf in subFolders:
            file_arr = os.listdir(sf)

            imageMatrixForOneSample = np.empty((pixelSize**2,0))
            darkImage = file_arr[0]
            theDarkFile = os.path.join(sf, darkImage)
            darkImageMatrix = cv2.imread(theDarkFile, cv2.IMREAD_GRAYSCALE)
            for img in file_arr[1:]:
                theFile = os.path.join(sf , img)
                imgMatrix = cv2.imread(theFile , cv2.IMREAD_GRAYSCALE)
                darkReductedImage = cv2.subtract(imgMatrix,darkImageMatrix)

                # Apply Adaptive Wiener Filter (Self Coded)
                filtered_image = wiener_filter(darkReductedImage, 3)

                superImage = getSuperPixel(filtered_image, pixelSize)
                columnMatrix = superImage.reshape(pixelSize**2, 1)
                imageMatrixForOneSample = np.hstack((imageMatrixForOneSample, columnMatrix))

            imageMatrixforOneLevel = np.vstack((imageMatrixforOneLevel,imageMatrixForOneSample)) 

        mainMatrix = np.vstack((mainMatrix,imageMatrixforOneLevel))

    return mainMatrix

def getSuperPixel(img, size):
    
    shp = img.shape
    hight = shp[0]
    width = shp[1]

    superImage = np.zeros( (round(hight/size) , round(width/size)) )

    for i in range(0,round(width/size)):
        for j in range(0,round(hight/size)):
            piece = img[i*size:size*(i+2)-size, j*size:size*(j+2)-size]
            superImage[i,j] = np.sum(piece)/(size**2)

    return superImage

def yMatrix():
    yMatrix = np.zeros(((pixelSize**2)*numberOfSamples*numOfLevels,1))
    for i in range((pixelSize**2)*numberOfSamples):
        yMatrix[i*(pixelSize**2)*numberOfSamples:(i+1)*(pixelSize**2)*numberOfSamples,0] = i

    return yMatrix

def makingDataSet():
    inputMatrix = MakeMainMatrix_DarkCurrentImage()
    outputMatrix = yMatrix()
    dataSet = np.append(inputMatrix, outputMatrix, axis = 1)
    print(dataSet)
    print(dataSet.shape)
    return dataSet

x = makingDataSet()
np.savetxt("Aflatoxin Reflectance Dark Current Reducted Wiener(3) Filtered Train Data Set (Super Pixel).csv", x, delimiter=',')
