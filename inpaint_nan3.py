import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.io import loadmat
from scipy.interpolate import CubicSpline
import pdb

def inpaint_nans(image):
    valid_mask = ~np.isnan(image)
    bscans = np.where(~valid_mask[-1,:])[0]
    for i in bscans:
        inpaint = image[max(np.where(valid_mask[:,i])[0]),i]
        image[-1,i] = inpaint
        valid_mask[-1,i] = True


    coords = np.array(np.nonzero(valid_mask)).T
    values = image[valid_mask]
    it = interpolate.LinearNDInterpolator(coords, values, fill_value=0)
    filled = it(list(np.ndindex(image.shape))).reshape(image.shape)

    return filled
