import json

import main
from main import plot_results
from workshop_utils import *
import numpy as np
import svm_tools
import os
from svm_tools import *

start_from_frame=50
until_frame=500
label_group="gaze_labels"
number_of_frames=until_frame-start_from_frame
subjects=os.listdir("subjects")[5:8]
average_angle_eyes=[]
angle_of_face=[]
for subject_num in subjects:
    print(subject_num)
    coordinates_dict=extract_coordinates_for_all_frames(0,start_from_frame,until_frame,"face_keypoints_2d",["left_eyelid_bottom_left","left_eyelid_bottom_right",
                                                                 "right_eyelid_bottom_left","right_eyelid_bottom_right",
                                                                 "left_eye_pupil","right_eye_pupil",
                                                                 "left_side_head","right_side_head",
                                                                 "nose_top","nose_bottom","chin"],
                                                            [36,40,47,45,68,69,0,16,27,33,8],subject_num)


    for i in range(number_of_frames):
        average_angle_eyes.append(get_angle_between_three_points(coordinates_dict["left_eyelid_bottom_left"][i][:2],coordinates_dict["left_eyelid_bottom_right"][i][:2],coordinates_dict["left_eye_pupil"][i][:2]))
        angle_of_face.append(get_angle_between_three_points(coordinates_dict["nose_bottom"][i][:2],coordinates_dict["nose_top"][i][:2],coordinates_dict["chin"][i][:2]))


"""
temp_X= [np.array(average_dist_between_eyelids).reshape(-1, 1),
     np.array(average_dist_of_pupils_from_center_of_eye).reshape(-1, 1),
     np.array(distance_of_nose_from_center_of_face).reshape(-1, 1)]
"""

temp_X= [np.array(average_angle_eyes).reshape(-1, 1),
     np.array(angle_of_face).reshape(-1, 1)]

X = np.concatenate(temp_X, axis=1)
labels_file = pd.read_csv("annotation/ep 1.csv")

"""
    gaze shift: 0
    face gaze:  1
    other gaze: 2
"""

#y  =[2*int(labels_file["OG."][i])+int(labels_file["FG."][i]) for i in range(start_from_frame,until_frame)]
labels = svm_tools.get_labels_from_file(file_path='ep 1.xlsx')
y=[]
for subject_num in subjects:
    svm_tools.subjects_dict[subject_num]
    y.extend([labels[svm_tools.subjects_dict[subject_num]]['gaze_labels'][i] for i in range(start_from_frame,until_frame)])

print(y)
y=svm_tools.convert_labels_to_ints(y,label_type=label_group)
#print(labels[1]['facial_exp_labels'][158], labels[1]['facial_exp_labels'][160])
test_points_indexes = np.random.choice([0, 1], size=len(X), p=[.75, .25])
X_train=[]
y_train=[]
X_test=[]
y_test=[]
for i,test_point in enumerate(test_points_indexes):
    if test_point:
        X_test.append(X[i])
        y_test.append(y[i])
    else:
        X_train.append(X[i])
        y_train.append(y[i])

linear_clf, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='linear')
rbf_clf, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='rbf')
poly_clf, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='poly') #TODO change back to poly
sig_clf, accuracy = run_svm_classifier(X_train, X_test, y_train, y_test, kernel='sigmoid')

main.plot_results([linear_clf],['Linear kernel'],X, y, "eye angel","face angel")
main.plot_results([sig_clf],[ 'Sigmoid kernel'],X, y, "eye angel","face angel")
main.plot_results([rbf_clf],[ 'RBF kernel'],X, y, "eye angel","face angel")
main.plot_results([poly_clf],['poly kernel'],X, y, "eye angel","face angel")

"""
average_dist_between_eyelids=[]
average_dist_of_pupils_from_center_of_eye=[]
distance_of_nose_from_center_of_face=[]
average_dist_between_eyelids.append((distance_between_2_points(coordinates_dict["eyelid_top_right"][i],
                                                                       coordinates_dict["eyelid_bottom_right"][i])+distance_between_2_points(coordinates_dict["eyelid_top_left"][i],
                                                                                                                                             coordinates_dict["eyelid_bottom_left"][i]))/(2))
        average_dist_of_pupils_from_center_of_eye.append(((distance_between_2_points(avg_2_points(coordinates_dict["right_eye_right"][i],
                                                                                                 coordinates_dict["right_eye_left"][i]),
                                                                                    coordinates_dict["right_eye_pupil"][i]))
                                                         +distance_between_2_points(avg_2_points(coordinates_dict["left_eye_right"][i],
                                                                                                 coordinates_dict["left_eye_left"][i]),
                                                                                    coordinates_dict["left_eye_pupil"][i]))/(2))

        #distance_of_nose_from_center_of_face.append(distance_between_2_points(avg_2_points(coordinates_dict["left_side_head"][i],
         #                                                                       coordinates_dict["right_side_head"][i])
          #                                                                      ,coordinates_dict["nose_top"][i])/(2*normalizing_factor))
          """

"""
    normalizing_factor_arr=[]
    for i in range(number_of_frames):
        normalizing_factor_arr.append(distance_between_2_points(coordinates_dict["nose_top"][i], coordinates_dict["nose_bottom"][i]))

    normalizing_factor=sum(normalizing_factor_arr) / len(normalizing_factor_arr)
"""