import numpy as np
import pandas as pd
import h5py

trainForecastPath = 'Data/ForecastDataforTraining_201712.csv'
measurementPath = 'Data/In_situMeasurementforTraining_201712.csv'
testForecastPath = 'Data/ForecastDataforTesting_201712.csv'
cityFile = 'Data/CityData.csv'
toFile = 'Data/METdata.h5'


def make_h5(trainForecastPath=trainForecastPath, testForecastPath=testForecastPath, measurementPath=measurementPath, toFile=toFile):
    """Load the data and save it as a h5 file fore fast future loading"""
    train = np.zeros((5, 20-3+1, 11, 548, 421))  # initialize an empty 5D tensor
    print('start processing traindata')
    with open(trainForecastPath) as trainfile:
        for index, line in enumerate(trainfile):
            #traindata format
            #xid,yid,date_id,hour,model,wind
            #1,1,1,3,1,13.8

            traindata = line.split(',')
            try:
                x = int(traindata[0])
                y = int(traindata[1])
                d = int(traindata[2])
                h = int(traindata[3])
                m = int(traindata[4])
                w = float(traindata[5])
                train[d-1, h-3, m-1, x-1, y-1] = w  # write values into tensor

                if index % 1000000 == 0:
                    print('%i lines has been processed' % (index))
            except ValueError:
                print("found line with datatype error! skip this line")
                continue

    print('start processing labeldata')
    with open(measurementPath) as labelfile:
        for index, line in enumerate(labelfile):
            #labeldata format
            #xid,yid,date_id,hour,wind
            #1,1,1,3,12.8
            labeldata = line.split(',')
            try:
                lx = int(labeldata[0])
                ly = int(labeldata[1])
                ld = int(labeldata[2])
                lh = int(labeldata[3])
                lw = float(labeldata[4])
                train[ld-1, lh-3, 10, lx-1, ly-1] = lw
                if index % 1000000 == 0:
                    print('%i lines has been processed' % (index))
            except ValueError:
                print("found line with datatype error! skip this line")
                continue

    test = np.zeros((5, 20-3+1, 10, 548, 421))
    print('start processing testdata')
    with open(testForecastPath) as testfile:
        for index, line in enumerate(testfile):
            #testdata format
            #xid,yid,date_id,hour,model,wind
            #1,1,1,3,1,13.8

            testdata = line.split(',')
            try:
                x = int(testdata[0])
                y = int(testdata[1])
                d = int(testdata[2])
                h = int(testdata[3])
                m = int(testdata[4])
                w = float(testdata[5])
                test[d-6, h-3, m-1, x-1, y-1] = w

                if index % 1000000 == 0:
                    print('%i lines has been processed' % (index))
            except ValueError:
                print("found line with datatype error! skip this line")
                continue

    # write numpy arrary tensor into h5 format
    h5f = h5py.File(toFile, 'w')
    h5f.create_dataset('train', data=train)
    h5f.create_dataset('test', data=test)
    h5f.close()


def getData(toFile=toFile, cityFile=cityFile):
    """Loads the data from the previously created h5"""
    print('Loading data from H5 file')
    citydata = pd.read_csv(cityFile)
    h5f = h5py.File(toFile, 'r')
    train = h5f['train'][:]
    test = h5f['test'][:]
    h5f.close()

    # At that point, the data has axes day*hour*model*x*y
    # The train data has one more 'model' than the test, which is the actual measurement
    return citydata, train, test


def splitTrainVal(train, nTrainDays, nValidationDays, useModels, test=False):
    """Splits the train data into one set for training of the predictor and the other to validate
        the prediction and the pathfindin"""
    if not test:
        useModels = useModels+[10]
    trainData = train[:nTrainDays, :, useModels, :, :]
    # Push the model axis to the end to be able to flatten the rest
    trainData = np.rollaxis(trainData, 2, 5)
    # Reshape the data to 2D in order to perform estimation
    trainData = trainData.reshape(-1, trainData.shape[-1])
    # Separate the true measurements from the forecast models
    x_train = trainData[:, 0:-1]
    y_train = trainData[:, -1]

    valData = train[nTrainDays:(nValidationDays+nTrainDays), :, useModels, :, :]
    valData = np.rollaxis(valData, 2, 5)
    valData = valData.reshape(-1, trainData.shape[-1])
    x_val = valData[:, 0:-1]
    y_val = valData[:, -1]

    return x_train, x_val, y_train, y_val
