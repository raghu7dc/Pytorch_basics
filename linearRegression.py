import torch 
import torch.nn as nn 
import numpy as np 
from sklearn import datasets

features, labels = datasets.load_diabetes(return_X_y=True)

print(features, labels )