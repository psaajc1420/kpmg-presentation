import time
import pandas as pd
import numpy as np

from scipy import interp
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def create_model():

    # Creating the Neural Network
    classifier = Sequential()

    # Creating the input layer
    classifier.add(Dense(units = 100, activation = "relu", kernel_initializer = "uniform", input_dim = 30))

    # Creating all the hidden layers
    classifier.add(Dense(units = 100, activation = "relu", kernel_initializer = "uniform"))
    classifier.add(Dense(units = 1, activation = "sigmoid", kernel_initializer = "uniform"))

    # Specify the optimization and loss functions for the 
    classifier.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["accuracy"])

    return classifier


def cross_validate(X, y, n_splits=10, visuals=True):

    # Applying K-Fold Cross Validation 
    cv = KFold(n_splits=n_splits)

    scores = np.zeros(n_splits)
    for i, (train, test) in enumerate(cv.split(X, y)):

        # Fitting the Neural Network to the training set
        classifier.fit(X[train], y[train], batch_size = 200, epochs = 30)

        # Predicting the test set using fitted model
        y_pred = classifier.predict(X[test])
        y_pred = (y_pred > 0.5)

        if visuals:
            pass

        cm, accuracy, precision, recall, f1_score = metrics(y[test], y_pred, output=False)

        scores[i] = accuracy

    print("Average Accucracy: {:.2f}", scores.mean())


def metrics(y_test, y_pred, output=True):

    # Creating the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn)/(tn + tp + fn + fp)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1_score = 2*precision*recall/(recall + precision)

    if output:
        print(cm)
        print('Accuracy: {:.2f}'.format(accuracy))
        print('Precsion: {:.2f}'.format(precision))
        print('Recall: {:.2f}'.format(recall))
        print('F1 score: {:.2f}'.format(f1_score))

    return cm, accuracy, precision, recall, f1_score

def roc_curve():
    pass

def tune_parameters():
    pass

if __name__ == "__main__":

    # Attain the dataset
    dataset = pd.read_csv("~/.datasets/creditcard.csv")

    # Get the index of the last column
    n = dataset.shape[1] - 1

    # Attain the features and labels of the dataset
    X = dataset.iloc[:, :n].values
    y = dataset.iloc[:, n].values

    # Splitting the data set in training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Applying feature scaling to the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    # Create the Neural Network model
    classifier = create_model()

    # Fitting the Neural Network to the training set
    start_train = time.time()
    classifier.fit(X_train, y_train, batch_size = 200, epochs = 30)
    train_time = (time.time() - start_train)
    print("Training time: {}".format(train_time))

    # Predicting the test set using fitted model
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    metrics(y_test, y_pred, output=True)

    # Running K-Fold Cross validation
    start_kfold = time.time()
    cross_validate(X, y, n_splits=10, visuals=True)
    kfold_time = time.time() - start_kfold
    print("K-Fold training time: {}".format(kfold_time))





