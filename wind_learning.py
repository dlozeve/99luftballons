import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


def makePrediction(x_train, x_val, y_train, model, modelName, red):
    """Fits the model on the training data and use it to make a prediction
    on both the train and validation data."""
    if red is not None:
        print('Performing PCA for', red, 'principal components')
        pca = PCA(red)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        x_val = pca.transform(x_val)
        del pca
        print('Done.')
        print()

    print('Fitting model', modelName)
    model.fit(x_train, y_train > 15)

    pred_val = model.predict_proba(x_val)[:, 1]
    pred_train = model.predict_proba(x_train)[:, 1]
    print('Done.')
    return pred_train, pred_val


def formatPrediction(pred_val, y_val, nValidationDays):
    print("Formatting the prediction")
    predMatrix = pred_val.reshape(nValidationDays, 18, 548, 421)
    if y_val is not None:
        trueMatrix = y_val.reshape(nValidationDays, 18, 548, 421)
    else:
        # Case when we are using the test data
        trueMatrix = None
    return predMatrix, trueMatrix


def evaluatePrediction(pred_train, pred_val, y_train, y_val, nValidationDays, thresh):
    # Generate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val >= 15, pred_val > thresh).ravel()
    print('False negative rate (most important metric):', fn/(fn+tn))
    print('False positive rate:', fp/(fp+tp))

    # Plot the actual wind speed against the predicted probability of being over 15
    indices = np.random.choice(pred_train.shape[0], 5000)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)

    ax1.scatter(pred_train[indices], y_train[indices])
    ax2.scatter(pred_val[indices], y_val[indices])

    ax1.axhline(15, color='r')
    ax2.axhline(15, color='r')

    ax1.axvline(thresh, color='r')
    ax2.axvline(thresh, color='r')

    ax1.set_xlabel('Predicted probability (training data)')
    ax2.set_xlabel('Predicted probability (validation data)')

    ax1.set_ylabel('Actual wind speed')

    ax1.text(0.25, 5, 'TN', color='black', size=30, horizontalalignment='center')
    ax1.text(0.25, 25, 'FN', color='r', size=30, horizontalalignment='center')
    ax1.text(0.75, 25, 'TP', color='black', size=30, horizontalalignment='center')
    ax1.text(0.75, 5, 'FP', color='black', size=30, horizontalalignment='center')
    plt.draw()

    # Show the prediction difference on a map for the first hour of the first validation day
    predMatrix = pred_val.reshape(nValidationDays, 18, 548, 421, 1)
    trueMatrix = y_val.reshape(nValidationDays, 18, 548, 421)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 20))

    ax1.imshow(trueMatrix[0, 0, :, :].T > 15)
    ax2.imshow(predMatrix[0, 0, :, :, 0].T > thresh)

    ax1.set_xlabel('True data')
    ax2.set_xlabel('Prediction')

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(np.logical_xor(trueMatrix[0, 0, :, :].T > 15, predMatrix[0, 0, :, :, 0].T > thresh))
    ax.set_xlabel('Yellow: wrong prediction, purple: right prediction')
    plt.draw()
