#use this as alternative of preprocess.m
#ignore the steps that are not useful for preprocessData function
import octSpectralisReader as rd
import numpy as np
import preprocessData as pd
import matplotlib.pyplot as plt
import cv2

[header, BScanHeader, slo, BScans] = rd.octSpectralisReader(r'C:\Users\Steve\Documents\LAB\Lab_Python\OCTMatTool\example.vol')
header['angle'] = 0

#initialize options:
options = {
    'preproc_params': {'normalize':2, 'filter':0,'filter_kernel':[1,1,3], 'flatten':1,'fast_rpe':0},
    'types':'hc',
    'img_rows':128,
    'img_cols':128,

    'segfile': 'example.mat', #need to convert this part
    'train': 0,
    'segs': 10,
    'crop': 0,
    'dplot':0, 
    'org':1,
    'flatvol':1
}
preproc_params = options['preproc_params']
preproc_params['retinadetector_type'] = options['types']

scanner_type = 'spectralis'
[img_vol,retina_mask,bds,shifts] = pd.preprocessData(BScans,header,preproc_params,scanner_type,1)
plt.imshow(img_vol[:,:,0], cmap= 'gray')
plt.plot
plt.show()


