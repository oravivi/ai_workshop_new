#Importing the necessary packages and libaries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np

# eyebrows & eyes
## left eyebrow:
#### center 19
#### left edge 17
#### right edge 21

## right eyebrow:
#### center 24
#### left edge 22
#### right edge 26

## left eye
#### upper eyelib 37, 38
#### lower eyelib 40, 41
#### center 68
#### left edge 36
#### right edge 39

## right eye
#### upper eyelib 43, 44
#### lower eyelib 46, 47
#### center 69
#### left edge 42
#### right edge 45

## Distance between the center of the eyebrow and the eye

## distance between the center and the edges of each eyebrow
