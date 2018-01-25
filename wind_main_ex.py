import wind_pathfinding as pfind
import wind_handling as handling
import wind_learning as learning
import wind_model as model
import wind_algo as algo

import os.path
import matplotlib.pyplot as plt
import pandas as pd

############ Parameters

# 'validation' if testing the model, 'test' if using test data to generate a submission
dataType = 'validation'
# file to save the submission
outFile = 'Output/sub.csv'

# Number of days used for training and validation of the forecast.
# Must sum at most to 5
nTrainDays = 2
nValidationDays = 2
# The input forecast models to use for the prediction. Must have values between 0 and 9 (/!\ 10 is the true data)
useModels = [0, 1, 2, 3, 4, 5]

# The model to use for the classification
###model = XGBClassifier(n_estimators=100, n_jobs=-1)
pred_model = model.model
modelName = model.modelName
# None if not PCA, number of components otherwise
red = model.red

# The threshold to use when evaluating the prediction and for the pathfinding
evalThreshold = model.evalThreshold
# The algorithm to use for pathfinding. See wind_pathfinding for inputs
algo = algo.stupidAlgo

############ Get the data and format it for prediction
if not os.path.isfile(handling.toFile):
    handling.make_h5()

citydata, train, test = handling.getData()

if dataType == 'validation':
    print('Splitting the data in train and validation sets')
    x_train, x_val, y_train, y_val = handling.splitTrainVal(train, nTrainDays, nValidationDays, useModels)
elif dataType == 'test':
    print("Reshaping the data for prediction")
    x_train, _, y_train, _ = handling.splitTrainVal(train, nTrainDays, 0, useModels)
    x_test, _, _, _ = handling.splitTrainVal(test, 5, 0, useModels)
del train, test

############ Fit the estimator and perform the prediction. Evaluate graphically and with some metrics
if dataType == 'validation':
    pred_train, pred_val = learning.makePrediction(x_train, x_val, y_train, pred_model, modelName, red)
    learning.evaluatePrediction(pred_train, pred_val, y_train, y_val, nValidationDays, evalThreshold)
    predMatrix, trueMatrix = learning.formatPrediction(pred_val, y_val, nValidationDays)
elif dataType == 'test':
    _, pred_test = learning.makePrediction(x_train, x_test, y_train, pred_model, modelName, red)
    predMatrix, _ = learning.formatPrediction(pred_test, None, 5)
    trueMatrix = None

############ Use the prediction to perform the pathfinding and a few evaluations. Output one path 
############ for verification as well as a complete submission
print("Getting sample path for verification")
start, stop, samplePath = pfind.getOnePath(algo, 1, 0, citydata, predMatrix)
pd.Series(samplePath).to_csv('Output/temp_checkpath.csv')
print('Getting full submission')
pfind.showFullPath(start, stop, samplePath, trueMatrix[0, :, :, :] > 15)
submission = pfind.getSubmission(algo, citydata, predMatrix, nValidationDays ,dataType, trueMatrix)
pd.DataFrame(submission).to_csv(outFile, header=False)

plt.show()
