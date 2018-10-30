import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

def transforming_data(dataset):

    # Get the index of the last column
    n = dataset.shape[1] - 1

    # Attain the features and labels of the dataset
    X = dataset.iloc[:, :n].values
    y = dataset.iloc[:, n].values

    # Splitting the data set in training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Applying feature scaling to 
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    return  X_train, X_test, y_train, y_test

def train(X_train, y_train):

    # Creating the Neural Network
    classifier = Sequential()

    # Creating the input layer
    classifier.add(Dense(units = 100, activation = "relu", kernel_initializer = "uniform", input_dim = 30))

    # Creating all the hidden layers
    classifier.add(Dense(units = 50, activation = "relu", kernel_initializer = "uniform"))
    classifier.add(Dense(units = 20, activation = "relu", kernel_initializer = "uniform"))
    classifier.add(Dense(units = 5, activation = "relu", kernel_initializer = "uniform"))
    classifier.add(Dense(units = 1, activation = "sigmoid", kernel_initializer = "uniform"))

    # Specify the optimization and loss functions for the 
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

    # Fitting the Neural Network to the training set
    classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

    return classifier

def test(X_test, y_test, classifier):
    
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0,5)
    metrics(y_test, y_pred)

def metrics(y_test, y_pred):
    # Predicting the test set using fitted model
    
    # Creating the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, tp, fn, fp = cm.ravel()
    accuracy = (tp + tn)/(tn + tp + fn + fp)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1_score = 2*accuracy*recall/(recall + precision)

    return cm, accuracy, precision, recall, f1_score


if __name__ == "__main__":

    # Attain the dataset
    dataset = pd.read_csv("~/.datasets/creditcard.csv")

    # Grab the transformed features and labels split up into test and training sets
    X_train, X_test, y_train, y_test = transforming_data(dataset)

    # Train the Neural Net 
    classifier = train(X_train, y_train)

    # Test the accuracy of the model by predicting on the test set 
    test(X_test, y_test, classifier)

    









