#Importing the necessary packages and libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
import workshop_utils


# distance between the center of the eyebrow and the eye
def left_eyebrow_center_to_eye_dist():
    return

# distance between the center and the edges of each eyebrow

# distance between the eyebrows

# distance between the edges line and the upper lip (bottom)

# distance between the edges line and the lower lip (top)

if __name__ == '__main__':
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
             'mouth_lower_lip_bottom_1',
             'mouth_lower_lip_bottom_2',
             'mouth_lower_lip_bottom_3',
             'mouth_lower_lip_top_1',
             'mouth_lower_lip_top_2',
             'mouth_lower_lip_top_3',
             'mouth_left_edge',
             'mouth_right_edge']
    numbers = [19, 17, 21, 24, 22, 26, 37, 38, 40, 41, 68, 36, 39, 43, 44, 46,
               47, 69, 42, 45, 61, 62, 63, 50, 51, 52, 56, 57, 58, 48, 54]

    infant_dict = workshop_utils.extract_coordinates_for_all_frames(0, 2, 'face_keypoints_2d' ,names, numbers)
    adult_dict = workshop_utils.extract_coordinates_for_all_frames(1, 2, 'face_keypoints_2d', names, numbers)

    features_matrix = []