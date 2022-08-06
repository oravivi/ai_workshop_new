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

def extract_coordinates_for_all_frames(person_id,number_of_frames,names,points):
    return_dict = {}
    for name in names:
        return_dict[name] = []
    for i in range(number_of_frames):
        file_path=".\output_json\clip_00000000000"+str(i)+"_keypoints.json"
        f = open(file_path)
        data = json.load(f)
        for i,name in enumerate(names):
            return_dict[name].append(data['people'][person_id]['face_keypoints_2d'][points[i]])

    return return_dict
    # returns JSON object as
# a dictionary


def get_angle_between_vectors(vector1, vector2):
    """
    :param vector1:
    :param vector2:
    :return: Angle between vector1 and vector2 in RADIANS
    """
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    angle = np.arccos(dot_product)
    return angle


