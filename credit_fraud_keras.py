import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import interp
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
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


def cross_validate(X, y, classifier, n_splits=10, visuals=True):

    # Applying K-Fold Cross Validation 
    cv = StratifiedKFold(n_splits=n_splits)

    scores = np.zeros(n_splits)
    precisions = np.zeros(n_splits)
    recalls = np.zeros(n_splits)
    f1_scores = np.zeros(n_splits)

    if visuals:
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
    
    for i, (train, test) in enumerate(cv.split(X, y)):

        # Fitting the Neural Network to the training set
        classifier.fit(X[train], y[train], batch_size = 200, epochs = 30, verbose=0)

        # Predicting the test set using fitted model
        y_probs = classifier.predict(X[test], batch_size=200)
        y_classes = y_probs.argmax(axis=-1)
        y_pred = (y_probs > 0.5) 
        
        print(len(y_probs))
        print(len(y[test]))

        print(y_probs)
        print(y_classes)
        print(y[test])
        print(y_pred)

        if visuals:
            fpr, tpr, _ =  roc_curve(y[test], y_probs)
            print(fpr)
            print(tpr)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            plt.plot(fpr, tpr, lw=1, alpha=0.4,
                     label="ROC fold {:d} (AUC = {:0.2f})".format(i, roc_auc))

        _, accuracy, precision, recall, f1_score = metrics(y[test], y_classes, output=False)

        scores[i] = accuracy
        precisions[i] = precision
        recalls[i] = recall
        f1_scores[i] = f1_score

        break

    if visuals:
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                    label='Chance', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC for Neural Net with Folds')
        plt.legend(loc="lower right")
        plt.savefig("ROC_folds.png")
            

    print("Average Accucracy: {:.2f}".format(scores.mean()))
    print('Average Precsion: {:.2f}'.format(precisions.mean()))
    print('Average Recall: {:.2f}'.format(recalls.mean()))
    print('Average F1 score: {:.2f}'.format(f1_score.mean()))


def metrics(y_test, y_pred, output=True):

    # Creating the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn)/(tn + tp + fn + fp)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1_score = 2*precision*recall/(recall + precision)

    if output:
        print()
        print(cm)
        print('Accuracy: {:.2f}'.format(accuracy))
        print('Precsion: {:.2f}'.format(precision))
        print('Recall: {:.2f}'.format(recall))
        print('F1 score: {:.2f}'.format(f1_score))

    return cm, accuracy, precision, recall, f1_score

def tune_parameters():
    pass

if __name__ == "__main__":

    if sys.argv[1] == "nogpu":
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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

    # # Fitting the Neural Network to the training set
    # start_train = time.time()
    # classifier.fit(X_train, y_train, batch_size = 200, epochs = 30)
    # train_time = (time.time() - start_train)
    # print("Training time: {}".format(train_time))

    # # Predicting the test set using fitted model
    # y_pred = classifier.predict(X_test)
    # y_pred = (y_pred > 0.5)
    
    # metrics(y_test, y_pred, output=True)

    if sys.argv[1] == "gpu":
        # Running K-Fold Cross validation
        start_kfold = time.time()
        cross_validate(X, y, classifier, n_splits=5, visuals=True)
        kfold_time = time.time() - start_kfold
        print("K-Fold training time: {}".format(kfold_time))





