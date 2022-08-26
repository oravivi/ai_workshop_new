#Importing the necessary packages and libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from workshop_utils import *
from svm_tools import *


if __name__ == '__main__':
    # get relevant dict for all frames
    names = ['chin_bottom_left',
             'chin_bottom_middle',
             'chin_bottom_right',
             'mustache_left_edge',
             'mustache_middle',
             'mustache_right_edge',
             'outer_lip_left_corner',
             'outer_upper_lip_1',
             'outer_upper_lip_2',
             'outer_upper_lip_3',
             'outer_upper_lip_4',
             'outer_upper_lip_5',
             'outer_lip_right_corner',
             'outer_bottom_lip_1',
             'outer_bottom_lip_2',
             'outer_bottom_lip_3',
             'outer_bottom_lip_4',
             'outer_bottom_lip_5',
             'inner_lip_left_corner',
             'inner_upper_lip_1',
             'inner_upper_lip_2',
             'inner_upper_lip_3',
             'inner_lip_right_corner',
             'inner_bottom_lip_1',
             'inner_bottom_lip_2',
             'inner_bottom_lip_3'
             ]
    mouth_features = {"right inner corner" : [63, 64, 65],
                      "right outer corner" : [53, 54, 55],
                      "left inner corner" : [61, 60, 67],
                      "left outer corner" : [49, 48, 59]}

    # numbers = [60, 61, 62, 63, 64, 65, 66, 67, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 7, 8, 9, 31, 33, 35]
    numbers = [7, 8, 9, 31, 33, 35, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]

    start_from_frame = 30
    until_frame = 3000
    coordinates_dict = extract_coordinates_for_all_frames(person_id=0,
                                                          start_from_frame=start_from_frame,
                                                          until_frame=until_frame,
                                                          body_part="face_keypoints_2d",
                                                          names=names,
                                                          points=numbers,
                                                          subject="611_3m")

    mouth_right_inner_corner_angles = []
    mouth_right_outer_corner_angles = []
    mouth_left_inner_corner_angles = []
    mouth_left_outer_corner_angles = []
    for i in range(2970):
        mouth_right_inner_corner_angles.append(get_angle_between_three_points(
            (coordinates_dict['inner_upper_lip_3'][i][0], coordinates_dict['inner_upper_lip_3'][i][1]),
            (coordinates_dict['inner_lip_right_corner'][i][0], coordinates_dict['inner_lip_right_corner'][i][1]),
            (coordinates_dict['inner_bottom_lip_1'][i][0], coordinates_dict['inner_bottom_lip_1'][i][1])))

        mouth_right_outer_corner_angles.append(get_angle_between_three_points((coordinates_dict['outer_upper_lip_5'][i][0], coordinates_dict['outer_upper_lip_5'][i][1]),
                                                                              (coordinates_dict['outer_lip_right_corner'][i][0], coordinates_dict['outer_lip_right_corner'][i][1]),
                                                                              (coordinates_dict['outer_bottom_lip_1'][i][0], coordinates_dict['outer_bottom_lip_1'][i][1])))

        mouth_left_inner_corner_angles.append(get_angle_between_three_points((coordinates_dict['inner_upper_lip_1'][i][0], coordinates_dict['inner_upper_lip_1'][i][1]),
                                                                             (coordinates_dict['inner_lip_left_corner'][i][0], coordinates_dict['inner_lip_left_corner'][i][1]),
                                                                             (coordinates_dict['inner_bottom_lip_3'][i][0], coordinates_dict['inner_bottom_lip_3'][i][1])))

        mouth_left_outer_corner_angles.append(get_angle_between_three_points((coordinates_dict['outer_upper_lip_1'][i][0], coordinates_dict['outer_upper_lip_1'][i][1]),
                                                                             (coordinates_dict['outer_lip_left_corner'][i][0], coordinates_dict['outer_lip_left_corner'][i][1]),
                                                                             (coordinates_dict['outer_bottom_lip_5'][i][0], coordinates_dict['outer_bottom_lip_5'][i][1])))
    temp_X = [np.array(mouth_right_inner_corner_angles).reshape(-1, 1),
              np.array(mouth_right_outer_corner_angles).reshape(-1, 1),
              np.array(mouth_left_inner_corner_angles).reshape(-1, 1),
              np.array(mouth_left_outer_corner_angles).reshape(-1, 1)]

    # X = np.concatenate(temp_X, axis=1)
    # X = np.transpose(X)
    infant_features_matrix = []
    for feature in temp_X:
        infant_features_matrix.append(feature)
    infant_features_matrix = np.concatenate(infant_features_matrix, axis=1)

    labels = get_labels_from_file(file_path='ep 1.xlsx')
    y = [labels[1]['facial_exp_labels'][i] for i in range(start_from_frame, 3000)]
    X_train, X_test, y_train, y_test = split_data(infant_features_matrix, y, train_ratio=0.2)
    print(X_train)
    classifier, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='linear')
    classifier, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='rbf')
    classifier, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='poly')
    classifier, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='sigmoid')














    """
    labels_file = pd.read_csv("subjects/annotation/ep 1.csv")

    labels = svm_tools.get_labels_from_file(file_path='ep 1.xlsx')
    y = [labels[5]['verbal_labels'][i] for i in range(start_from_frame, until_frame)]

    test_points_indexes = np.random.choice([0, 1], size=len(X), p=[.85, .15])
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

    models = []
    titles = []

    C = 10
    clf_lin = svm.SVC(C=C, decision_function_shape='ovo')
    clf_lin.set_params(kernel='linear').fit(X_train, y_train)
    y_prediction = []
    for x in X_test:
        y_prediction.append(clf_lin.predict(x.reshape(1, -1)))
    print(sklearn.metrics.accuracy_score(y_test, y_prediction))
    models.append(clf_lin)
    titles.append("linear kernel")
    """





