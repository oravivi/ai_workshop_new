#Importing the necessary packages and libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from workshop_utils import *
from svm_tools import *
from facial_expressions import use_coordinates_directly


def get_feature_matrices(sub_no, frames_to_skip, frames_num):
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
             'inner_bottom_lip_3',
             'right_eyelid_1',
             'right_eyelid_2',
             'right_eyelid_3',
             'left_eyelid_1',
             'left_eyelid_2',
             'left_eyelid_3',
             'left_eyebrow_1',
             'left_eyebrow_2',
             'left_eyebrow_3',
             'left_eyebrow_4',
             'left_eyebrow_5',
             'right_eyebrow_1',
             'right_eyebrow_2',
             'right_eyebrow_3',
             'right_eyebrow_4',
             'right_eyebrow_5',
             ]

    numbers = [7, 8, 9, 31, 33, 35, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
               44, 45, 46, 37, 36, 41, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    frames_num = frames_num
    total_frames = frames_num-frames_to_skip
    infant_dict = extract_coordinates_for_all_frames(person_id=0, start_from_frame=frames_to_skip, until_frame=frames_num, body_part='face_keypoints_2d' ,names=names, points=numbers, subject=sub_no)
    #adult_dict = extract_coordinates_for_all_frames(1, 2, 'face_keypoints_2d', names, numbers)
    infant_features_matrix = []
    adult_features_matrix = []

    #infant_features_matrix, adult_features_matrix = extract_features_from_coordinates(infant_dict, adult_dict, total_frames)
    infant_features_matrix = use_coords(infant_dict, total_frames, sub_no)
    # infant_features_matrix = np.concatenate(infant_features_matrix, axis=1)
    #adult_features_matrix = np.concatenate(adult_features_matrix, axis=1)
    return infant_features_matrix

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
             'inner_bottom_lip_3',
             'right_eyelid_1',
             'right_eyelid_2',
             'right_eyelid_3',
             'left_eyelid_1',
             'left_eyelid_2',
             'left_eyelid_3',
             'left_eyebrow_1',
             'left_eyebrow_2',
             'left_eyebrow_3',
             'left_eyebrow_4',
             'left_eyebrow_5',
             'right_eyebrow_1',
             'right_eyebrow_2',
             'right_eyebrow_3',
             'right_eyebrow_4',
             'right_eyebrow_5',
             ]
    mouth_features = {"right inner corner" : [63, 64, 65],
                      "right outer corner" : [53, 54, 55],
                      "left inner corner" : [61, 60, 67],
                      "left outer corner" : [49, 48, 59]}

    # numbers = [60, 61, 62, 63, 64, 65, 66, 67, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 7, 8, 9, 31, 33, 35]
    numbers = [7, 8, 9, 31, 33, 35, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
               44, 45, 46, 37, 36, 41, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]

    start_from_frame = 30
    until_frame = 3400
    # coordinates_dict = extract_coordinates_for_all_frames(person_id=0,
    #                                                       start_from_frame=start_from_frame,
    #                                                       until_frame=until_frame,
    #                                                       body_part="face_keypoints_2d",
    #                                                       names=names,
    #                                                       points=numbers,
    #                                                       subject="3649_6m")

    mouth_right_inner_corner_angles = []
    mouth_right_outer_corner_angles = []
    mouth_left_inner_corner_angles = []
    mouth_left_outer_corner_angles = []
    right_eye_angles = []
    left_eye_angles = []
    right_eye_features = []
    left_eye_features = []
    mouth_right_inner_corner_x = []
    mouth_right_inner_corner_y = []
    mouth_left_inner_corner_x = []
    mouth_left_inner_corner_y = []
    mustache_right_edge_x = []
    mustache_right_edge_y = []
    outer_upper_lip_3_x = []
    outer_upper_lip_3_y = []
    outer_bottom_lip_3_x = []
    outer_bottom_lip_3_y = []
    chin_bottom_left_x = []
    chin_bottom_left_y = []
    chin_bottom_right_x = []
    chin_bottom_right_y = []

    left_eyebrow_1_x = []
    left_eyebrow_1_y = []
    left_eyebrow_2_x = []
    left_eyebrow_2_y = []
    left_eyebrow_3_x = []
    left_eyebrow_3_y = []
    left_eyebrow_4_x = []
    left_eyebrow_4_y = []
    left_eyebrow_5_x = []
    left_eyebrow_5_y = []

    right_eyebrow_1_x = []
    right_eyebrow_1_y = []
    right_eyebrow_2_x = []
    right_eyebrow_2_y = []
    right_eyebrow_3_x = []
    right_eyebrow_3_y = []
    right_eyebrow_4_x = []
    right_eyebrow_4_y = []
    right_eyebrow_5_x = []
    right_eyebrow_5_y = []
    lip_distances = []

    def use_coords(coords_dict, frames_num, subject_name):

        coordinates_dict = extract_coordinates_for_all_frames(person_id=0,
                                                              start_from_frame=start_from_frame,
                                                              until_frame=until_frame,
                                                              body_part="face_keypoints_2d",
                                                              names=names,
                                                              points=numbers,
                                                              subject=subject_name)

        for i in range(frames_num):
            mouth_right_inner_corner_angles.append(get_angle_between_three_points(
                (coordinates_dict['inner_upper_lip_3'][i][0], coordinates_dict['inner_upper_lip_3'][i][1]),
                (coordinates_dict['inner_lip_right_corner'][i][0], coordinates_dict['inner_lip_right_corner'][i][1]),
                (coordinates_dict['inner_bottom_lip_1'][i][0], coordinates_dict['inner_bottom_lip_1'][i][1])))

            mouth_right_inner_corner_x.append(coordinates_dict['inner_lip_right_corner'][i][0])
            mouth_right_inner_corner_y.append(coordinates_dict['inner_lip_right_corner'][i][1])
            mouth_left_inner_corner_x.append(coordinates_dict['inner_lip_left_corner'][i][0])
            mouth_left_inner_corner_y.append(coordinates_dict['inner_lip_left_corner'][i][1])
            mustache_right_edge_x.append(coordinates_dict['mustache_right_edge'][i][0])
            mustache_right_edge_y.append(coordinates_dict['mustache_right_edge'][i][1])
            outer_upper_lip_3_x.append(coordinates_dict['outer_upper_lip_3'][i][0])
            outer_upper_lip_3_y.append(coordinates_dict['outer_upper_lip_3'][i][1])
            outer_bottom_lip_3_x.append(coordinates_dict['outer_bottom_lip_3'][i][0])
            outer_bottom_lip_3_y.append(coordinates_dict['outer_bottom_lip_3'][i][1])
            chin_bottom_left_x.append(coordinates_dict['chin_bottom_left'][i][0])
            chin_bottom_left_y.append(coordinates_dict['chin_bottom_left'][i][1])
            chin_bottom_right_x.append(coordinates_dict['chin_bottom_right'][i][0])
            chin_bottom_right_y.append(coordinates_dict['chin_bottom_right'][i][1])
            left_eyebrow_1_x.append(coordinates_dict['left_eyebrow_1'][i][0])
            left_eyebrow_1_y.append(coordinates_dict['left_eyebrow_1'][i][1])
            left_eyebrow_2_x.append(coordinates_dict['left_eyebrow_1'][i][0])
            left_eyebrow_2_y.append(coordinates_dict['left_eyebrow_1'][i][1])
            left_eyebrow_3_x.append(coordinates_dict['left_eyebrow_1'][i][0])
            left_eyebrow_3_y.append(coordinates_dict['left_eyebrow_1'][i][1])
            left_eyebrow_4_x.append(coordinates_dict['left_eyebrow_1'][i][0])
            left_eyebrow_4_y.append(coordinates_dict['left_eyebrow_1'][i][1])
            left_eyebrow_5_x.append(coordinates_dict['left_eyebrow_1'][i][0])
            left_eyebrow_5_y.append(coordinates_dict['left_eyebrow_1'][i][1])

            right_eyebrow_1_x.append(coordinates_dict['right_eyebrow_1'][i][0])
            right_eyebrow_1_y.append(coordinates_dict['right_eyebrow_1'][i][1])
            right_eyebrow_2_x.append(coordinates_dict['right_eyebrow_1'][i][0])
            right_eyebrow_2_y.append(coordinates_dict['right_eyebrow_1'][i][1])
            right_eyebrow_3_x.append(coordinates_dict['right_eyebrow_1'][i][0])
            right_eyebrow_3_y.append(coordinates_dict['right_eyebrow_1'][i][1])
            right_eyebrow_4_x.append(coordinates_dict['right_eyebrow_1'][i][0])
            right_eyebrow_4_y.append(coordinates_dict['right_eyebrow_1'][i][1])
            right_eyebrow_5_x.append(coordinates_dict['right_eyebrow_1'][i][0])
            right_eyebrow_5_y.append(coordinates_dict['right_eyebrow_1'][i][1])
            lip_distances.append((distance_between_2_points((outer_upper_lip_3_x[i], outer_upper_lip_3_y[i]),
                                                            (outer_bottom_lip_3_x[i], outer_bottom_lip_3_y[i]))))


            mouth_right_outer_corner_angles.append(get_angle_between_three_points(
                (coordinates_dict['outer_upper_lip_5'][i][0], coordinates_dict['outer_upper_lip_5'][i][1]),
                (coordinates_dict['outer_lip_right_corner'][i][0], coordinates_dict['outer_lip_right_corner'][i][1]),
                (coordinates_dict['outer_bottom_lip_1'][i][0], coordinates_dict['outer_bottom_lip_1'][i][1])))

            mouth_left_inner_corner_angles.append(get_angle_between_three_points(
                (coordinates_dict['inner_upper_lip_1'][i][0], coordinates_dict['inner_upper_lip_1'][i][1]),
                (coordinates_dict['inner_lip_left_corner'][i][0], coordinates_dict['inner_lip_left_corner'][i][1]),
                (coordinates_dict['inner_bottom_lip_3'][i][0], coordinates_dict['inner_bottom_lip_3'][i][1])))

            mouth_left_outer_corner_angles.append(get_angle_between_three_points(
                (coordinates_dict['outer_upper_lip_1'][i][0], coordinates_dict['outer_upper_lip_1'][i][1]),
                (coordinates_dict['outer_lip_left_corner'][i][0], coordinates_dict['outer_lip_left_corner'][i][1]),
                (coordinates_dict['outer_bottom_lip_5'][i][0], coordinates_dict['outer_bottom_lip_5'][i][1])))

            right_eye_angles.append(get_angle_between_three_points(
                (coordinates_dict['right_eyelid_1'][i][0], coordinates_dict['right_eyelid_1'][i][1]),
                (coordinates_dict['right_eyelid_2'][i][0], coordinates_dict['right_eyelid_2'][i][1]),
                (coordinates_dict['right_eyelid_3'][i][0], coordinates_dict['right_eyelid_3'][i][1])
            ))

            left_eye_angles.append(get_angle_between_three_points(
                (coordinates_dict['left_eyelid_1'][i][0], coordinates_dict['left_eyelid_1'][i][1]),
                (coordinates_dict['left_eyelid_2'][i][0], coordinates_dict['left_eyelid_2'][i][1]),
                (coordinates_dict['left_eyelid_3'][i][0], coordinates_dict['left_eyelid_3'][i][1])
            ))

            # right_eye_features.append(((mouth_right_inner_corner_angles[i] + mouth_right_outer_corner_angles[i]) / 2) * right_eye_angles[i])
            # left_eye_features.append(((mouth_left_inner_corner_angles[i] + mouth_left_outer_corner_angles[i]) / 2) * left_eye_angles[i])


        temp_X = [np.array(mouth_right_inner_corner_angles).reshape(-1, 1),
                  np.array(mouth_right_outer_corner_angles).reshape(-1, 1),
                  np.array(mouth_left_inner_corner_angles).reshape(-1, 1),
                  np.array(mouth_left_outer_corner_angles).reshape(-1, 1),
                  np.array(right_eye_angles).reshape(-1, 1),
                  np.array(left_eye_angles).reshape(-1, 1),
                  ]
        # temp_X = [np.array(outer_upper_lip_3_x).reshape(-1, 1),
        #           np.array(outer_upper_lip_3_y).reshape(-1, 1),
        #           np.array(outer_bottom_lip_3_x).reshape(-1, 1),
        #           np.array(outer_bottom_lip_3_y).reshape(-1, 1),
        #           np.array(mustache_right_edge_x).reshape(-1, 1),
        #           np.array(mustache_right_edge_y).reshape(-1, 1),
        #           np.array(mouth_left_inner_corner_x).reshape(-1, 1),
        #           np.array(mouth_left_inner_corner_y).reshape(-1, 1),
        #           np.array(mouth_right_inner_corner_x).reshape(-1, 1),
        #           np.array(mouth_right_inner_corner_y).reshape(-1, 1),
        #           np.array(chin_bottom_left_x).reshape(-1, 1),
        #           np.array(chin_bottom_left_y).reshape(-1, 1),
        #           np.array(chin_bottom_right_x).reshape(-1, 1),
        #           np.array(chin_bottom_right_y).reshape(-1, 1),
        #           np.array(left_eyebrow_1_x).reshape(-1, 1),
        #           np.array(left_eyebrow_1_y).reshape(-1, 1),
        #           np.array(left_eyebrow_2_x).reshape(-1, 1),
        #           np.array(left_eyebrow_2_y).reshape(-1, 1),
        #           np.array(left_eyebrow_3_x).reshape(-1, 1),
        #           np.array(left_eyebrow_3_y).reshape(-1, 1),
        #           np.array(left_eyebrow_4_x).reshape(-1, 1),
        #           np.array(left_eyebrow_4_y).reshape(-1, 1),
        #           np.array(left_eyebrow_5_x).reshape(-1, 1),
        #           np.array(left_eyebrow_5_y).reshape(-1, 1),
        #           np.array(right_eyebrow_1_x).reshape(-1, 1),
        #           np.array(right_eyebrow_1_y).reshape(-1, 1),
        #           np.array(right_eyebrow_2_x).reshape(-1, 1),
        #           np.array(right_eyebrow_2_y).reshape(-1, 1),
        #           np.array(right_eyebrow_3_x).reshape(-1, 1),
        #           np.array(right_eyebrow_3_y).reshape(-1, 1),
        #           np.array(right_eyebrow_4_x).reshape(-1, 1),
        #           np.array(right_eyebrow_4_y).reshape(-1, 1),
        #           np.array(right_eyebrow_5_x).reshape(-1, 1),
        #           np.array(right_eyebrow_5_y).reshape(-1, 1)
        #           ]

        X = np.concatenate(temp_X, axis=1)
        # X = np.transpose(X)
        infant_features_matrix = []
        for feature in temp_X:
            infant_features_matrix.append(feature)
        infant_features_matrix = np.concatenate(infant_features_matrix, axis=1)

        return infant_features_matrix

    subjects = os.listdir("subjects")[0:3]
    random.shuffle(subjects)

    frames_to_skip = 30
    frame_num = 3400
    number_of_frames = frame_num - frames_to_skip
    y = []
    infant_x = 0
    for sub_no_from_file in subjects:
        converted_sub_no_labels = subjects_dict[sub_no_from_file]
        labels = get_labels_from_file(file_path='ep 1.xlsx')
        # y = [labels[20]['verbal_labels'][i] for i in range(start_from_frame, 3400)]
        y.extend([labels[converted_sub_no_labels]['verbal_labels'][i] for i in range(start_from_frame, 3400)])

        if (isinstance(infant_x, int)):
            infant_x = get_feature_matrices(frames_num=frame_num, frames_to_skip=frames_to_skip, sub_no=sub_no_from_file)
        else:
            infant_x=np.vstack((infant_x,get_feature_matrices(frames_num=frame_num, frames_to_skip=frames_to_skip, sub_no=sub_no_from_file)))
    infant_x_2d = reduce_dim(infant_x)
    # X_train, X_test, y_train, y_test = split_data(infant_x_2d, y, train_ratio=0.2)
    X_train, X_test, y_train, y_test = split_data_not_random(infant_x_2d, y, 2, 1, number_of_frames)
    print(X_train)
    classifiers = []
    classifiers.append(run_svm_classifier(X_train, X_test, y_train, y_test, kernel='linear')[0])
    classifiers.append(run_svm_classifier(X_train, X_test, y_train, y_test, kernel='rbf')[0])
    classifiers.append(run_svm_classifier(X_train, X_test, y_train, y_test, kernel='poly')[0])
    classifiers.append(run_svm_classifier(X_train, X_test, y_train, y_test, kernel='sigmoid')[0])
    titles = ['linear', 'rbf', 'poly', 'sigmoid']
    labels_for_plot = convert_labels_to_ints(y=y, label_type='verbal_labels')
    print(np.histogram(labels_for_plot))
    plot_results_2(infant_x_2d, labels_for_plot, classifiers, titles)
    print(np.histogram(labels_for_plot))













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





