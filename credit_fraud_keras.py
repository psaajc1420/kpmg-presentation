import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# Attain the dataset
dataset = pd.read_csv("~/.datasets/creditcard.csv")

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

# Creating the Neural Network
classifier = Sequential()

# Creating the input layer
classifier.add(Dense(units = 100, activation = "relu", kernel_initializer = "uniform", input_dim = 30))

# Creating all the hidden layers
classifier.add(Dense(units = 100, activation = "relu", kernel_initializer = "uniform"))
classifier.add(Dense(units = 1, activation = "sigmoid", kernel_initializer = "uniform"))

# Specify the optimization and loss functions for the 
classifier.compile(optimizer = "rmsprop", loss = "binary_crossentropy", metrics = ["accuracy"])

# Fitting the Neural Network to the training set
classifier.fit(X_train, y_train, batch_size = 200, epochs = 30)

# Predicting the test set using fitted model
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Creating the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn)/(tn + tp + fn + fp)
precision = tp/(tp + fp)
recall = tp/(tp + fn)
f1_score = 2*precision*recall/(recall + precision)

print('Accuracy: {:.2f}'.format(accuracy))
print('Precsion: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 score: {:.2f}'.format(f1_score))
