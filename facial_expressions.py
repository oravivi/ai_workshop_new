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


# angle between the mouth edges line and the upper lip (bottom)
def mouth_edges_to_lower_lip_angle(coordinates_dict, frames_num):
    angles = []
    for i in range(frames_num):
        dist = get_angle_between_three_points(
            (coordinates_dict['mouth_left_edge'][i][0], coordinates_dict['mouth_left_edge'][i][1]),
            (coordinates_dict['mouth_lower_lip_bottom_2'][i][0], coordinates_dict['mouth_lower_lip_bottom_2'][i][1]),
            (coordinates_dict['mouth_right_edge'][i][0],coordinates_dict['mouth_right_edge'][i][1]))
        angles.append(dist)
    return convert_row_to_column(angles)


def mouth_left_angle(coordinates_dict, frames_num):
    angles = []
    for i in range(frames_num):
        dist = get_angle_between_three_points(
            (coordinates_dict['mouth_upper_lip_top_2'][i][0], coordinates_dict['mouth_upper_lip_top_2'][i][1]),
            (coordinates_dict['mouth_left_edge'][i][0], coordinates_dict['mouth_left_edge'][i][1]),
            (coordinates_dict['mouth_lower_lip_bottom_2'][i][0], coordinates_dict['mouth_lower_lip_bottom_2'][i][1]))
        angles.append(dist)
    return convert_row_to_column(angles)


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
    adult_features_matrix = []
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

def get_feature_matrices(sub_no, frames_to_skip, frames_num):
    # get relevant dict for all frames
    names = ['left_eyebrow_center',
             'left_eyebrow_left_edge',
             'left_eyebrow_right_edge',
             'right_eyebrow_center',
             'right_eyebrow_left_edge',
             'right_eyebrow_right_edge',
             'left_eye_upper_eyelid_1',
             'left_eye_upper_eyelid_2',
             'left_eye_lower_eyelid_1',
             'left_eye_lower_eyelid_2',
             'left_eye_center',
             'left_eye_left_edge',
             'left_eye_right_edge',
             'right_eye_upper_eyelid_1',
             'right_eye_upper_eyelid_2',
             'right_eye_lower_eyelid_1',
             'right_eye_lower_eyelid_2',
             'right_eye_center',
             'right_eye_left_edge',
             'right_eye_right_edge',
             'mouth_upper_lip_bottom_1',
             'mouth_upper_lip_bottom_2',
             'mouth_upper_lip_bottom_3',
             'mouth_upper_lip_top_1',
             'mouth_upper_lip_top_2',
             'mouth_upper_lip_top_3',
             'mouth_lower_lip_top_1',
             'mouth_lower_lip_top_2',
             'mouth_lower_lip_top_3',
             'mouth_lower_lip_bottom_1',
             'mouth_lower_lip_bottom_2',
             'mouth_lower_lip_bottom_3',
             'mouth_left_edge',
             'mouth_right_edge']
    numbers = [19, 17, 21, 24, 22, 26, 37, 38, 40, 41, 68, 36, 39,
               43, 44, 46, 47, 69, 42, 45, 61, 62, 63, 50, 51, 52, 65, 66, 67, 56, 57, 58, 48, 54]
    frames_num = frames_num
    total_frames = frames_num-frames_to_skip
    infant_dict = extract_coordinates_for_all_frames(person_id=0, start_from_frame=frames_to_skip, until_frame=frames_num, body_part='face_keypoints_2d' ,names=names, points=numbers, subject=sub_no)
    #adult_dict = extract_coordinates_for_all_frames(1, 2, 'face_keypoints_2d', names, numbers)
    infant_features_matrix = []
    adult_features_matrix = []

    #infant_features_matrix, adult_features_matrix = extract_features_from_coordinates(infant_dict, adult_dict, total_frames)
    infant_features_matrix = extract_features_from_coordinates(infant_dict, total_frames)
    infant_features_matrix = np.concatenate(infant_features_matrix, axis=1)
    #adult_features_matrix = np.concatenate(adult_features_matrix, axis=1)
    return infant_features_matrix



if __name__ == '__main__':
    frames_to_skip=30
    frame_num=3000 #TODO use the number of frames in the directory
    sub_no_from_file = '611_3m'
    converted_sub_no_labels = 1 #TODO use dict to convert from the file name to sub_no
    infant_x = get_feature_matrices(frames_num=frame_num, frames_to_skip=frames_to_skip, sub_no=sub_no_from_file)
    labels = get_labels_from_file(file_path='ep 1.xlsx')
    y = [labels[converted_sub_no_labels]['facial_exp_labels'][i] for i in range(frames_to_skip, frame_num)]
    X_train, X_test, y_train, y_test = split_data(infant_x, y, train_ratio=0.2)
    linear_clf, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='linear')
    #rbf_clf, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='rbf')
    #poly_clf, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='linear') #TODO change back to poly
    #sig_clf, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='sigmoid')
    y_nums = convert_labels_to_ints(y, label_type='facial_exp_labels')
    infant_x_2d = reduce_dim(infant_x)
    #plot_results(infant_x, y_nums, classifiers=(linear_clf, rbf_clf, poly_clf, sig_clf),titles=['Linear kernel', 'RBF kernel', 'Polynomial kernel', 'Sigmoid kernel'])
    plot_results_2(infant_x, y_nums, models=[linear_clf, linear_clf, linear_clf,linear_clf],titles=['Linear kernel', 'Linear kernel','Linear kernel', 'Linear kernel'])




