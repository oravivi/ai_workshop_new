import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn import svm, datasets
import itertools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import random
import json
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay

subjects_dict={"611_3m" :1,
               "611_6m" :2,
               "3447_3m" :3,
               "3447_6m" :4,
               "409_3m" :5,
               "409_6m" :6,
               "2529_3m" :7,
               "2529_6m" :8,
               "2730_3m" :9,
               "2730_6m" :10,
               "2831_3m" :11,
               "2831_6m" :12,
               "3548_3m" :13,
               "3548_6m" :14,
               "3750_3m" :15,
               "3750_6m" :16,
               "4051_3m" :17,
               "4051_6m" :18,
               "3649_3m" :19,
               "3649_6m" :20,
               "4555_3m" :21,
               "4555_6m" :22,
               "6267_3m" :23,
               "6267_6m" :24,
               "6569_3m" :25,
               "6569_6m" :26,
               "6064_3m" :27,
               "6064_6m" :28,
               "6166_3m" :29,
               "6166_6m" :30,
               "4656_3m" :31,
               "4656_6m" :32,
               "7275_3m" :33,
               "7275_6m" :34,
               "5458_3m" :35,
               "5458_6m" :36,
               "7477_3m" :37,
               "7477_6m" :38,
               "7678_3m" :39,
               "7678_6m" :40

               }


# 'X' the matrix with the extracted features for each classification
# a row in X represents a specific feature, a column in X represents a specific frame
# y - a vector of labels
def split_data(X, y, train_ratio=0.2):
    test_points_indexes = np.random.choice([0, 1], size=len(X), p=[1-train_ratio, train_ratio])
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i, test_point in enumerate(test_points_indexes):
        if test_point:
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
    return X_train, X_test, y_train, y_test

def split_data_not_random(X, y, number_of_train_videos,number_of_test_videos,number_of_frames_per_video):
    test_points_indexes = np.concatenate((np.zeros(number_of_train_videos*number_of_frames_per_video), np.ones(number_of_test_videos*number_of_frames_per_video)), axis=0)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i, test_point in enumerate(test_points_indexes):
        if test_point:
            X_test.append(X[i])
            y_test.append(y[i])
        else:
            X_train.append(X[i])
            y_train.append(y[i])
    return X_train, X_test, y_train, y_test

def get_facial_exp_row(row):
    NE, FR, SM, OF = row['NE.'], row['FR.'], row['SM.'], row['OF.']
    if NE==1:
        return 'NE'
    elif FR==1:
        return 'FR'
    elif SM==1:
        return 'SM'
    elif OF==1:
        return 'OF'
    else:
        print("invalid label", "sub no = " + row["sub_no"])


def get_verbal_row(row):
    SP, NS, SL = row['SP.'], row['NSP.'], row['SL.']
    if SP==1:
        return 'SP'
    elif NS==1:
        return 'NS'
    elif SL==1:
        return 'SL'
    else:
        print("invalid label", "sub no = " + row["sub_no"])


def get_gaze_row(row):
    FG, GS, OG = row['FG.'], row['GS.'], row['OG.']
    if FG==1:
        return 'FG'
    elif GS==1:
        return 'GS'
    elif OG==1:
        return 'OG'
    else:
        print("invalid label", "sub no = " + row["sub_no"])


def get_hands_row(row):
    PH, RP, IR, RES = row['PH.'], row['RP.'], row['IR.'], row['RES.']
    if PH==1:
        return 'PH'
    elif RP==1:
        return 'RP'
    elif IR==1:
        return 'IR'
    elif RES==1:
        return 'RES'
    else:
        print("invalid label", "sub no = " + row["sub_no"])


def get_labels_from_file(file_path='ep 1.xlsx'):
    # verbal_labels = ['SP', 'NS', 'SL']
    # gaze_labels = ['FG', 'GS', 'OG']
    # facial_exp_labels = ['NE', 'FR', 'SM', 'OF']
    # hands_labels = ['PH', 'RP', 'IR', 'RT', 'HD']
    labels_data = {}
    for i in range(40):
        subject_labels = {'verbal_labels':[],
                             'gaze_labels':[],
                             'facial_exp_labels':[],
                             'hands_labels':[]
                                     }
        labels_data[i+1] = subject_labels
    print(labels_data)
    labels_from_excel = pd.read_excel(file_path)
    rows = labels_from_excel.iterrows()
    for i, row in rows:
        sub_no = row['sub_no']
        count = row['Count']
        facial_exp = get_facial_exp_row(row)
        verbal = get_verbal_row(row)
        gaze = get_gaze_row(row)
        hands = get_hands_row(row)

        # facial expressions
        labels_to_add = [facial_exp for i in range(count)]
        new_list = list(itertools.chain(labels_data[sub_no]['facial_exp_labels'], labels_to_add))
        labels_data[sub_no]['facial_exp_labels'] = new_list

        # verbal
        labels_to_add = [verbal for i in range(count)]
        new_list = list(itertools.chain(labels_data[sub_no]['verbal_labels'], labels_to_add))
        labels_data[sub_no]['verbal_labels'] = new_list

        # gaze
        labels_to_add = [gaze for i in range(count)]
        new_list = list(itertools.chain(labels_data[sub_no]['gaze_labels'], labels_to_add))
        labels_data[sub_no]['gaze_labels'] = new_list

        # hands
        labels_to_add = [hands for i in range(count)]
        new_list = list(itertools.chain(labels_data[sub_no]['hands_labels'], labels_to_add))
        labels_data[sub_no]['hands_labels'] = new_list
    #print(labels_data)
    return labels_data


# 'X' the matrix with the extracted features for each classification
# a row in X represents a specific feature, a column in X represents a specific frame
# y - a vector of labels
def run_svm_classifier(X_test, X_train, y_test ,y_train, kernel):
    C=10
    if kernel=='linear':
        classifier = svm.SVC(C=C, kernel='linear', decision_function_shape='ovo').fit(X_train, y_train)
    elif kernel=='rbf':
        classifier = svm.SVC(kernel='rbf', gamma=1, C=C, decision_function_shape='ovo').fit(X_train, y_train)
    elif kernel=='poly':
        classifier = svm.SVC(kernel='poly', degree=3, C=C, decision_function_shape='ovo').fit(X_train, y_train)
    elif kernel=='sigmoid':
        classifier = svm.SVC(kernel='sigmoid', C=C, decision_function_shape='ovo').fit(X_train, y_train)
    y_prediction = []
    #classifier.set_params(kernel='linear').fit(X_train, y_train)
    for x in X_test:
        y_prediction.append(classifier.predict(x.reshape(1, -1)))
    accuracy = sklearn.metrics.accuracy_score(y_test, y_prediction)
    print(kernel, accuracy)
    return classifier, accuracy


def convert_labels_to_ints(y, label_type='facial_exp_labels'):
    if label_type=='facial_exp_labels':
        labels_to_num_facial_exp = {'NE':0, 'FR':1, 'SM':2, 'OF':3}
    elif label_type == 'verbal_labels':
        labels_to_num_facial_exp = {'SP':0, 'NS':1, 'SL':2}
    elif label_type == 'gaze_labels':
        labels_to_num_facial_exp = {'GS': 0, 'FG': 1, 'OG': 2}
    else:
        print ("does not support label type")
    return [labels_to_num_facial_exp[label] for label in y]


def reduce_dim(X):
    X_2d = TSNE(n_components=2, learning_rate='auto', init = 'random', perplexity = 30).fit_transform(X)
    return X_2d

def plot_results_2(X,y, models, titles):
    '''
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    '''
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]

    for clf, title, ax in zip(models, titles, sub.flatten()):
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax
        )
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
    plt.show()




def plot_results(X, y, classifiers, titles =['Linear kernel', 'RBF kernel', 'Polynomial kernel', 'Sigmoid kernel']):
    # stepsize in the mesh, it alters the accuracy of the plotprint
    # to better understand it, just play with the value, change it and print it
    h = .01
    # create the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # create the title that will be shown on the plot
    titles = titles

    for i, clf in enumerate(classifiers):
        # defines how many plots: 2 rows, 2columns=> leading to 4 plots
        plt.subplot(2, 2, i + 1)  # i+1 is the index
        # space between plots
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = convert_labels_to_ints(Z, label_type='facial_exp_labels')
        Z = np.array(Z)
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.PuBuGn, alpha=0.7)
        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.PuBuGn, edgecolors='grey')
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])
        print("done")
    plt.show()


#labels = get_labels_from_file(file_path='.\ep 1.xlsx')
#print(labels[1]['facial_exp_labels'][158], labels[1]['facial_exp_labels'][160])