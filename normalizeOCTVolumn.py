#function that normalize the OCT volumn
#different intensity normalizaation methods are shown below
import numpy as np
from numpy import matlib as mb
from scipy import integrate 
from scipy import ndimage as nd
import matplotlib.pyplot as plt


def normalizeOCTVolumn(*arg):
    in_vol = arg[0]
    method = arg[1]
    header = arg[2]
    retina_bs = []
    downsample = False

    #downsample data to fixed resolution:
    if downsample: #not reached below, implementation might have problem
        re_res = np.array([4,10])
        old_res = 1000* [header['ScaleZ'], header['ScaleX']]
        old_res = np.array(old_res) 
        re_res = np.amax(re_res, old_res)
        sc =old_res/re_res
        new_size = np.round(in_vol[:,:,1].shape * sc)
        #image resize functino need to be convert into python
    else:
        in_vol_rs = in_vol
        re_res = 1000* [header['ScaleZ'], header['ScaleX']]
        
    #median filter kernel
    hsz = 27
    h = [2*np.round(hsz/re_res[0]+1), 1]
    out_vol = np.zeros(in_vol.shape) 
    

    #Only method 4 is fully implemented
    if method == 1:
        #median filter standard deviation normalization
        #stdv  nanstd funtion in matlab need to be transfered
        for j in range(in_vol.shape[2]):
            #med = medfilt2  python transfer
            #out_vol[:,:,j] = in_vol[:,:,j]/nanstd... 
            quit()
    elif method == 2:
        #median filter contrast stretching normalization
        if isinstance(in_vol, int):
            #convert the class of in_vol into double
            quit()
        else:
            mv = 1
            maxOffset =0.05*mv 
            
            for j in range(in_vol.shape[2]):
                #med = medfilt2 
                ms = max(med) + maxOffset
                if ms > mv:
                    ms = mv
                #out_vol[:,:,j] = imadjust function need to transfer
    elif method == 3:
        qle3 = 0.3
        qle2 = 0.999
        #q1 = quantile()
        #q2 = quantile()

        for j in range(in_vol.shape[3]):
            s1 = in_vol[:,:,j]
            #q1 q2 quantile
            #out_vol[:,:,j] = imadjust
    elif method == 4: # method called in the example case
        #shadow removal and contrast
        in_vol = np.power(in_vol, 4)
        

        in_temp = np.flipud(in_vol)
        cs =  np.flipud(integrate.cumtrapz(in_temp, initial=0, axis =0)) #not sure if the cumtrapz is implemented correctly
        
        sigma = 15/(header['ScaleX']*1000)
        sigma = float(sigma)
        cs = nd.gaussian_filter(cs, (sigma,0,0) ,mode = 'mirror')
        
        temp  = np.flipud(np.power(in_vol,2))
        E = np.flipud(integrate.cumtrapz(temp, initial=0, axis= 0))
        E = E < 0.0001

        for i in range(in_vol.shape[2]):
            temp = np.all(E[:,:,i], axis = 1)
            ind  = np.argmax(temp.astype(int))
            
            cs[ind:,:,i] = mb.repmat(cs[ind,:,i],cs.shape[0]-ind, 1)
    
        out_vol = in_vol/cs 
        out_vol[np.isnan(out_vol)] = np.min(out_vol)
        out_vol[np.isinf(out_vol)] = np.min(out_vol)
        out_vol = np.power(out_vol, 0.25) 

    return out_vol 

            


        