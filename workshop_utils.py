import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn import svm, datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import random
import json

def extract_coordinates_for_all_frames(person_id,number_of_frames,body_part,names,points):
    return_dict = {}
    for name in names:
        return_dict[name] = []
    for i in range(number_of_frames):
        file_path=".\output_json\clip_00000000000"+str(i)+"_keypoints.json"
        f = open(file_path)
        data = json.load(f)
        for i,name in enumerate(names):
            point=(data['people'][person_id][body_part][points[i]*3],data['people'][person_id][body_part][points[i]*3+1],data['people'][person_id][body_part][points[i]*3+2])
            return_dict[name].append(point)

    return return_dict

def distance_between_2_points(point_a,point_b):
    x_dist=abs(point_a[0]-point_b[0])
    y_dist=abs(point_b[1]-point_b[1])
    return math.sqrt(x_dist**2+y_dist**2)

def avg_2_points(point_a,point_b):
    new_point=((point_a[0]+point_b[0])/2,(point_a[1]+point_b[1])/2,(point_a[2]+point_a[2]/2))
    return new_point


# a dictionary


def get_angle_between_three_points(point1, point2, point3):
    """
    :param point1: [x,y]
    :param point2: [x,y]
    :param point3: [x,y]
    :return: Angle between vector1 and vector2 in RADIANS
    """
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return angle



