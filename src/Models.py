import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

#For SVM implementation
from sklearn.svm import SVC

#For KNN implementation
from sklearn.neighbors import KNeighborsClassifier

#For LDA implementation
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#For Naive Bayes implementation
from sklearn.naive_bayes import GaussianNB

#For MLP implementation
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

import tensorflow as tf
from keras import datasets, layers, models


def SVM(X, Y):
    seed = 42
    
    # Reshape Y to make it one-dimensional
    Y = Y.ravel()
    print(np.shape(Y))

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

    # Create an SVM classifier
    clf_svm = SVC(random_state=seed)
    #clf_svm = SVC(kernel='linear')
    #clf_svm = SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    #    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    #   max_iter=-1, probability=False, random_state=42, shrinking=True, tol=0.001,
    #  verbose=False)   

    # Train the SVM classifier
    clf_svm.fit(x_train, y_train)
     
    # Testing the model
    # Make predictions on the test set
    y_pred = clf_svm.predict(x_test)
    #y_pred = np.argmax(y_preds,axis=1)
    #y_pred = np.argmax(y_preds)

    # Calculate and print the accuracy of the model
    print("SVM Accuracy :-- {:.2f}".format(accuracy_score(y_test, y_pred)*100) + "%")

    #Plot Confusion Matrix for the true and predicted labels
    cf = confusion_matrix(y_test, y_pred)
    #ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=clf_svm.classes_)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.suptitle('Confusion matrix of Gesture lables using SVM')
    plt.title("Accuracy of :-- {:.2f}".format(accuracy_score(y_test, y_pred)*100) + "%")
    plt.show()
    #plt.savefig('SVM_Pred_Confusion.png')

    # printing the report
    ##print(classification_report(y_test, y_pred))
    return cf


def KNN(X, Y):
    seed = 42
    
    # Reshape Y to make it one-dimensional
    Y = Y.ravel()
    print(np.shape(Y))

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

    # Create an KNN classifier
    clf_knn = KNeighborsClassifier(n_neighbors=3)

    # Train the SVM classifier
    clf_knn.fit(x_train, y_train)
     
    # Testing the model
    # Make predictions on the test set
    y_pred = clf_knn.predict(x_test)

    # Calculate and print the accuracy of the model
    print("KNN Accuracy :-- {:.2f}".format(accuracy_score(y_test, y_pred)*100) + "%")

    #Plot Confusion Matrix for the true and predicted labels
    cf = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=clf_knn.classes_)
    plt.suptitle('Confusion matrix of Gesture lables using KNN')
    plt.title("Accuracy of :-- {:.2f}".format(accuracy_score(y_test, y_pred)*100) + "%")
    plt.show()
    #plt.savefig('KNN_Pred_Confusion.png')
    return


def LDA(X, Y):
    seed = 42
    
    # Reshape Y to make it one-dimensional
    Y = Y.ravel()
    print(np.shape(Y))

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

    # Create an KNN classifier
    clf_lda = LinearDiscriminantAnalysis()

    # Train the SVM classifier
    clf_lda.fit(x_train, y_train)
     
    # Testing the model
    # Make predictions on the test set
    y_pred = clf_lda.predict(x_test)

    # Calculate and print the accuracy of the model
    print("LDA Accuracy :-- {:.2f}".format(accuracy_score(y_test, y_pred)*100) + "%")

    #Plot Confusion Matrix for the true and predicted labels
    cf = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=clf_lda.classes_)
    plt.suptitle('Confusion matrix of Gesture lables using LDA')
    plt.title("Accuracy of :-- {:.2f}".format(accuracy_score(y_test, y_pred)*100) + "%")
    plt.show()
    #plt.savefig('KNN_Pred_Confusion.png')

    return


def NB(X, Y):
    seed = 42
    
    # Reshape Y to make it one-dimensional
    Y = Y.ravel()
    print(np.shape(Y))

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

    # Create an KNN classifier
    clf_nb = GaussianNB()

    # Train the SVM classifier
    clf_nb.fit(x_train, y_train)
     
    # Testing the model
    # Make predictions on the test set
    y_pred = clf_nb.predict(x_test)

    # Calculate and print the accuracy of the model
    print("NB Accuracy :-- {:.2f}".format(accuracy_score(y_test, y_pred)*100) + "%")

    #Plot Confusion Matrix for the true and predicted labels
    cf = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=clf_nb.classes_)
    plt.suptitle('Confusion matrix of Gesture lables using Naive Bayes')
    plt.title("Accuracy of :-- {:.2f}".format(accuracy_score(y_test, y_pred)*100) + "%")
    plt.show()
    #plt.savefig('KNN_Pred_Confusion.png')

    return

def DNN(X, Y):
    # Reshape Y to make it one-dimensional
    Y = Y.ravel()
    print(np.shape(Y))

    # Parameters
    nGestExamples, featLen = X.shape
    num_labels = np.unique(Y)
    batch_size = 512 # It is the sample size of inputs to be processed at each training stage. 
    hidden_units = 512
    dropout = 0.45
    seed = 42

    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(Y)
    num_classes = len(label_encoder.classes_)

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

    # Normalize the input features
    #x_train = x_train / 255.0
    #x_test = x_test / 255.0

    # Convert the labels to one-hot encoded vectors
    y_train_OHE = np_utils.to_categorical(y_train, num_classes)
    y_test_OHE = np_utils.to_categorical(y_test, num_classes)

    # Create an MLP model
    model = Sequential()

    # Add dense layers
    model.add(Dense(hidden_units, activation='relu', input_shape=(featLen,)))
    model.add(Dropout(dropout))

    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(dropout))

    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train_OHE, epochs=10, batch_size=32, validation_data=(x_test, y_test_OHE))

    # Evaluate the model
    _, accuracy = model.evaluate(x_test, y_test_OHE)
    print("MLP Accuracy:-- {:.2f}".format(accuracy*100) + "%")

    y_pred_probab = model.predict(x_test) #.predict_classes(x_test)
    # #y_pred = model.call(x_test)
    # #y_pred1 = (model.predict(x_test)>0.5).astype(int) #(y_pred_probab > 0.5)
    # nTestSamples = x_test.shape[0]
    # y_pred = np.zeros((nTestSamples,), dtype=np.int64)
    # for n in range(nTestSamples):
    #     probab = y_pred_probab[n, :]
    #     maxProbab = np.max(probab)
    #     predIdx = np.where(probab == maxProbab)
    #     y_pred[n] = predIdx #np.where(probab == maxProbab) #y_pred1[n,:] == 1)
    
    y_pred = np.argmax(y_pred_probab,axis=1)

    cf = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred) #labels=model.classes_
    plt.suptitle('Confusion matrix of Gesture lables using MLP')
    plt.title("Accuracy of :-- {:.2f}".format(accuracy_score(y_test, y_pred)*100) + "%")
    plt.show()
    #plt.savefig('MLP_Pred_Confusion.png')
    return
    # # Parameters
    # nGestExamples, featLen = X.shape
    # num_labels = np.unique(Y)
    # batch_size = 512 # It is the sample size of inputs to be processed at each training stage. 
    # hidden_units = 512
    # dropout = 0.45

    # print("train_images shape: ", train_images.shape)
    # print("test_images shape: ", test_images.shape) 

    # # Using Sequential() to build layers one after another
    # model = Sequential()

    # # Hidden layer
    # model.add(Dense(hidden_units, input_dim=featLen))
    # model.add(Activation('relu'))
    # model.add(Dropout(dropout))

    # # Hidden layer
    # model.add(Dense(hidden_units))
    # model.add(Activation('relu'))
    # model.add(Dropout(dropout))

    # # Hidden layer
    # model.add(Dense(hidden_units))
    # model.add(Activation('relu'))
    # model.add(Dropout(dropout))

    # # Hidden layer
    # model.add(Dense(hidden_units))
    # model.add(Activation('relu'))
    # model.add(Dropout(dropout))

    # model.add(Dense(num_labels))

    # model.add(Activation('softmax'))
    # model.summary()