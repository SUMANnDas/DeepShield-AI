import torch
from torchvision.transforms import Normalize

def normalize_faces(x):
    return Normalize((.485,.456,.406),(.229,.224,.225))(x)
