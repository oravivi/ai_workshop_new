#Importing the necessary packages and libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from workshop_utils import *
from svm_tools import *


# distance between the center of the eyebrow and the eye
def left_eyebrow_center_to_eye_dist(coordinates_dict: dict, frames_num):
    distances = []
    for i in range(frames_num):
        dist = distance_between_2_points(
            (coordinates_dict['left_eyebrow_center'][i][0], coordinates_dict['left_eyebrow_center'][i][1]),
            (coordinates_dict['left_eye_center'][i][0], coordinates_dict['left_eye_center'][i][1]))
        distances.append(dist)
    return convert_row_to_column(distances)


def right_eyebrow_center_to_eye_dist(coordinates_dict: dict, frames_num):
    distances = []
    for i in range(frames_num):
        dist = distance_between_2_points(
            (coordinates_dict['right_eyebrow_center'][i][0], coordinates_dict['right_eyebrow_center'][i][1]),
            (coordinates_dict['right_eye_center'][i][0], coordinates_dict['right_eye_center'][i][1]))
        distances.append(dist)
    return convert_row_to_column(distances)


# distance between the center and the edges of each eyebrow
def left_eyebrow_edges_to_center_angle():
    pass


def right_eyebrow_edges_to_center_angle():
    pass


# distance between the eyebrows
def eyebrows_dist(coordinates_dict: dict, frames_num):
    distances = []
    for i in range(frames_num):
        dist = distance_between_2_points(
            (coordinates_dict['left_eyebrow_right_edge'][i][0], coordinates_dict['left_eyebrow_right_edge'][i][1]),
            (coordinates_dict['right_eyebrow_left_edge'][i][0], coordinates_dict['right_eyebrow_left_edge'][i][1]))
        distances.append(dist)
    return convert_row_to_column(distances)


# angle between the mouth edges line and the lower lip (bottom)
def mouth_edges_to_lower_lip_angle(coordinates_dict, frames_num):
    angles = []
    for i in range(frames_num):
        dist = get_angle_between_three_points(
            (coordinates_dict['mouth_left_edge'][i][0], coordinates_dict['mouth_left_edge'][i][1]),
            (coordinates_dict['mouth_lower_lip_bottom_2'][i][0], coordinates_dict['mouth_lower_lip_bottom_2'][i][1]),
            (coordinates_dict['mouth_right_edge'][i][0],coordinates_dict['mouth_right_edge'][i][1]))
        angles.append(dist)
    return convert_row_to_column(angles)

# mouth left angle
def mouth_left_angle(coordinates_dict, frames_num):
    angles = []
    for i in range(frames_num):
        dist = get_angle_between_three_points(
            (coordinates_dict['mouth_upper_lip_top_2'][i][0], coordinates_dict['mouth_upper_lip_top_2'][i][1]),
            (coordinates_dict['mouth_left_edge'][i][0], coordinates_dict['mouth_left_edge'][i][1]),
            (coordinates_dict['mouth_lower_lip_bottom_2'][i][0], coordinates_dict['mouth_lower_lip_bottom_2'][i][1]))
        angles.append(dist)
    return convert_row_to_column(angles)

# mouth right angle
def mouth_right_angle(coordinates_dict, frames_num):
    angles = []
    for i in range(frames_num):
        dist = get_angle_between_three_points(
            (coordinates_dict['mouth_upper_lip_top_2'][i][0], coordinates_dict['mouth_upper_lip_top_2'][i][1]),
            (coordinates_dict['mouth_right_edge'][i][0], coordinates_dict['mouth_right_edge'][i][1]),
            (coordinates_dict['mouth_lower_lip_bottom_2'][i][0], coordinates_dict['mouth_lower_lip_bottom_2'][i][1]))
        angles.append(dist)
    return convert_row_to_column(angles)


def extract_features_from_coordinates(infant_dict, frames_num):
    infant_features_matrix = []
    #adult_features_matrix = []
    #infant
    infant_features_matrix.append(left_eyebrow_center_to_eye_dist(infant_dict, frames_num))
    infant_features_matrix.append(right_eyebrow_center_to_eye_dist(infant_dict, frames_num))
    infant_features_matrix.append(mouth_edges_to_lower_lip_angle(infant_dict, frames_num))
    infant_features_matrix.append(mouth_right_angle(infant_dict, frames_num))
    infant_features_matrix.append(mouth_left_angle(infant_dict, frames_num))
    infant_features_matrix.append(eyebrows_dist(infant_dict, frames_num))
    #adult
    #adult_features_matrix.append(left_eyebrow_center_to_eye_dist(adult_dict, frames_num))
    #adult_features_matrix.append(right_eyebrow_center_to_eye_dist(adult_dict, frames_num))
    #adult_features_matrix.append(mouth_edges_to_lower_lip_angle(adult_dict, frames_num))
    return infant_features_matrix

def use_coordinates_directly(coordinates_dict, frames_num):
    print(coordinates_dict.keys())
    infant_features_matrix = []
    infant_features_matrix.append(convert_row_to_column([coordinates_dict['right_eye_right'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(convert_row_to_column([coordinates_dict['right_eye_right'][i][1] for i in range(frames_num)]))
    infant_features_matrix.append(convert_row_to_column([coordinates_dict['right_eye_left'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(convert_row_to_column([coordinates_dict['right_eye_left'][i][1] for i in range(frames_num)]))
    infant_features_matrix.append(convert_row_to_column([coordinates_dict['right_eye_top'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(convert_row_to_column([coordinates_dict['right_eye_top'][i][1] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['right_eye_bottom'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['right_eye_bottom'][i][1] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['right_eye_pupil'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['right_eye_pupil'][i][1] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['left_eye_right'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['left_eye_right'][i][1] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['right_eye_top'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['right_eye_top'][i][1] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['left_eye_left'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['left_eye_left'][i][1] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['left_eye_top'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['left_eye_top'][i][1] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['left_eye_bottom'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['left_eye_bottom'][i][1] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['left_eye_pupil'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['left_eye_pupil'][i][1] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['nose_bottom'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['nose_bottom'][i][1] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['nose_top'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['nose_top'][i][1] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['chin'][i][0] for i in range(frames_num)]))
    infant_features_matrix.append(
        convert_row_to_column([coordinates_dict['chin'][i][1] for i in range(frames_num)]))
    return infant_features_matrix
def get_feature_matrices(sub_no, frames_to_skip, frames_num):
    # get relevant dict for all frames
    names = ['right_eye_right',
             "right_eye_left",
             "right_eye_top",
             "right_eye_bottom",
             "right_eye_pupil",
             'left_eye_right',
             "left_eye_left",
             "left_eye_top",
             "left_eye_bottom",
             "left_eye_pupil",
             "nose_bottom",
             "nose_top",
             "chin"]
    numbers = [45, 42, 43, 47, 69, 39, 36, 38, 40, 68, 30, 27, 8]
    frames_num = frames_num
    total_frames = frames_num-frames_to_skip
    infant_dict = extract_coordinates_for_all_frames(person_id=0, start_from_frame=frames_to_skip, until_frame=frames_num, body_part='face_keypoints_2d' ,names=names, points=numbers, subject=sub_no)
    #adult_dict = extract_coordinates_for_all_frames(1, 2, 'face_keypoints_2d', names, numbers)
    infant_features_matrix = []
    adult_features_matrix = []

    #infant_features_matrix, adult_features_matrix = extract_features_from_coordinates(infant_dict, adult_dict, total_frames)
    infant_features_matrix = use_coordinates_directly(infant_dict, total_frames)
    infant_features_matrix = np.concatenate(infant_features_matrix, axis=1)
    #adult_features_matrix = np.concatenate(adult_features_matrix, axis=1)
    return infant_features_matrix



if __name__ == '__main__':
    label_group="gaze_labels"
    frames_to_skip=30
    frame_num=300 #TODO use the number of frames in the directory
    subjects = os.listdir("subjects")[0:2]
    y=[]
    infant_x=0
    for sub_no_from_file in subjects:
        print(sub_no_from_file)
        #sub_no_from_file = '611_3m'
        converted_sub_no_labels = subjects_dict[sub_no_from_file] #TODO use dict to convert from the file name to sub_no
        labels = get_labels_from_file(file_path='ep 1.xlsx')
        y.extend([labels[converted_sub_no_labels][label_group][i] for i in range(frames_to_skip, frame_num)])
        if (isinstance(infant_x, int)):
            infant_x = get_feature_matrices(frames_num=frame_num, frames_to_skip=frames_to_skip, sub_no=sub_no_from_file)
        else:
            infant_x=np.vstack((infant_x,get_feature_matrices(frames_num=frame_num, frames_to_skip=frames_to_skip, sub_no=sub_no_from_file)))
    print(len(y))
    print(type(infant_x),print(infant_x.shape))

    #X_train, X_test, y_train, y_test = split_data(infant_x, y, train_ratio=0.2)
    #linear_clf, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='linear')
    #infant_x_test_2d = reduce_dim(infant_x_test)
    infant_x_2d = reduce_dim(infant_x)
    X_train, X_test, y_train, y_test = split_data(infant_x_2d, y, train_ratio=0.5)
    linear_clf, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='linear')
    rbf_clf, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='rbf')
    poly_clf, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='poly') #TODO change back to poly
    sig_clf, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='sigmoid')
    #y_nums = convert_labels_to_ints(y, label_type=label_group)
    #plot_results(infant_x_2d, y_nums, classifiers=(linear_clf, rbf_clf, poly_clf, sig_clf),titles=['Linear kernel', 'RBF kernel', 'Polynomial kernel', 'Sigmoid kernel'])
    plot_results_2(infant_x_2d, y_nums, models=[linear_clf, rbf_clf, poly_clf, sig_clf],titles=['Linear kernel', 'RBF kernel','poly kernel', 'Sigmoid kernel'])




