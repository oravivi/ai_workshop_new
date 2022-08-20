import json
from main import plot_results
from workshop_utils import *
import numpy as np

start_from_frame=120
until_frame=145
number_of_frames=until_frame-start_from_frame
coordinates_dict=extract_coordinates_for_all_frames(0,start_from_frame,until_frame,"face_keypoints_2d",["eyelid_top_left","eyelid_bottom_left",
                                                             "eyelid_top_right","eyelid_bottom_right",
                                                             "right_eye_right","right_eye_left",
                                                             "left_eye_right","left_eye_left",
                                                             "right_eye_pupil","left_eye_pupil",
                                                             "left_side_head","right_side_head",
                                                             "nose_top"],
                                                        [37,41,44,46,45,42,39,36,69,68,0,16,27],"sub 611 ep1")


average_dist_between_eyelids=[]
average_dist_of_pupils_from_center_of_eye=[]
distance_of_nose_from_center_of_face=[]
for i in range(number_of_frames):
    average_dist_between_eyelids.append((distance_between_2_points(coordinates_dict["eyelid_top_right"][i],
                                                                   coordinates_dict["eyelid_bottom_right"][i])+distance_between_2_points(coordinates_dict["eyelid_top_left"][i],
                                                                                                                                         coordinates_dict["eyelid_bottom_left"][i]))/2)
    average_dist_of_pupils_from_center_of_eye.append(((distance_between_2_points(avg_2_points(coordinates_dict["right_eye_right"][i],
                                                                                             coordinates_dict["right_eye_left"][i]),
                                                                                coordinates_dict["right_eye_pupil"][i]))
                                                     +distance_between_2_points(avg_2_points(coordinates_dict["left_eye_right"][i],
                                                                                             coordinates_dict["left_eye_left"][i]),
                                                                                coordinates_dict["left_eye_pupil"][i]))/2)
    """
    distance_of_nose_from_center_of_face.append(distance_between_2_points(avg_2_points(coordinates_dict["left_side_head"][i],
                                                                            coordinates_dict["right_side_head"][i])
                                                                            ,coordinates_dict["nose_top"][i])/2)
          
    """                                                                  """
    temp_X= [np.array(average_dist_between_eyelids).reshape(-1, 1),
         np.array(average_dist_of_pupils_from_center_of_eye).reshape(-1, 1),
         np.array(distance_of_nose_from_center_of_face).reshape(-1, 1)]"""



temp_X = [np.array(average_dist_between_eyelids).reshape(-1, 1),
          np.array(average_dist_of_pupils_from_center_of_eye).reshape(-1, 1)]

X = np.concatenate(temp_X, axis=1)
labels_file = pd.read_csv("subjects/annotation/ep 1.csv")

"""
    gaze shift: 0
    face gaze:  1
    other gaze: 2
"""

y  =[2*int(labels_file["OG."][i])+int(labels_file["FG."][i]) for i in range(start_from_frame,until_frame)]
test_points_indexes = np.random.choice([0, 1], size=len(X), p=[.85, .15])
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

models = []
titles = []

C=10
clf_lin = svm.SVC(C=C)
clf_lin.set_params(kernel='linear').fit(X_train, y_train)
y_prediction=[]
for x in X_test:
    y_prediction.append(clf_lin.predict(x.reshape(1,-1)))
print (sklearn.metrics.accuracy_score(y_test,y_prediction))
models.append(clf_lin)
titles.append("linear kernel")

plot_results(models, titles, X, y,"angle","distance")

