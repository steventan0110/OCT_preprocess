#flatten the data by shifting A-scans up or down depending on the value in shifts
#extrapolation adds zeros to the top or bottom of the A-scan
import numpy as np 
import scipy as sp
from scipy import interpolate as ip
def retinaFlatten(img_data, shifts, interp):
    """
    Flattening by shifting A-scans up or down depending on the value in 'shifts'
    Extrapolation just adds zeros to the top or bottom of the A-scan
    Args:
    vol: image volume [img_rows, img_cols, Bscans]
    shifts: [img_rows, Bscans]
    interp: interp method. 'linear' or 'nearest'
    Return: 
    shiftImg: [img_rows, img_cols, Bscans]
    """
    img_rows, img_cols, B = img_data.shape
    types = img_data.dtype
    assert interp == 'linear' or interp == 'nearest'
    xx = np.arange(img_cols)
    yy = np.arange(img_rows)
    [X,Y] = np.meshgrid(xx,yy)
    Ym = Y[:,:,np.newaxis] + shifts[np.newaxis,:,:]
    shiftImg = np.zeros(img_data.shape)
    for i in range(B):
        f = ip.RegularGridInterpolator((yy,xx),img_data[:,:,i],bounds_error=False,method=interp)
        z = np.stack((Ym[:,:,i],X),-1)
        shiftImg[:,:,i] = f(z) 
    shiftImg[np.isnan(shiftImg)] = 0
    return shiftImg.astype(types)
    
