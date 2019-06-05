import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd 
from scipy import signal as sg
from scipy import interpolate as ip
import inpaint_nan3 as inan
#get parameter function that check the parameter 


#main function starts here:
def retinaDetector(img_vol,header,paramSet,doplots):
    def getParams(paramSet):
        params = dict()
        for x in ['default','spectralis','hc','mme']:
            if paramSet == x: 
                #Originally for the spectralis
                params['sigma_lat'] = 16.67
                params['sigma_ax'] = 11.6
                params['distconst'] = 96.68
                params['sigma_lat_ilm'] = 55.56
                params['sigma_lat_isos'] = 55.56
                params['sigma_lat_bm'] = 111.13
                params['maxdist'] = 386.73 # ~100 pixels in spectralis
                params['bsc_indep'] = False
                return params
        if paramSet == 'dme':
            params['sigma_lat'] = 16.67
            params['sigma_ax'] = 11.6
            params['distconst'] = 96.68
            params['sigma_lat_ilm'] = 55.56
            params['sigma_lat_isos'] =55.56
            params['sigma_lat_bm'] = 111.13
            params['maxdist'] = 386.73 # ~100 pixels in spectralis
            params['bsc_indep'] = False
        elif paramSet == 'cirrus':
            params['sigma_lat'] = 2*16.67
            params['sigma_ax'] = 0.5*11.6
            params['distconst'] = 96.68
            params['sigma_lat_ilm'] = 55.56
            params['sigma_lat_isos'] =55.56
            params['sigma_lat_bm'] = 111.13
            params['maxdist'] = 386.73 # ~100 pixels in spectralis
            params['bsc_indep'] = True
        elif paramSet == 'cirrus_sm':
            params['sigma_lat'] = 2*16.67
            params['sigma_ax'] = 0.5*11.6
            params['distconst'] = 96.68
            params['sigma_lat_ilm'] = 55.56
            params['sigma_lat_isos'] =55.56
            params['sigma_lat_bm'] = 200
            params['maxdist'] = 386.73 # ~100 pixels in spectralis
            params['bsc_indep'] = True
        elif paramSet == 'rp':
            params['sigma_lat'] = 2*16.67
            params['sigma_ax'] = 0.5*11.6
            params['distconst'] = 50
            params['sigma_lat_ilm'] = 200
            params['sigma_lat_isos'] = 300
            params['sigma_lat_bm'] = 200
            params['maxdist'] = 386.73 # ~100 pixels in spectralis
            params['bsc_indep'] = True
        elif paramSet == 'phantom':
            params['sigma_lat'] = 5
            params['sigma_ax'] = 5
            params['distconst'] = 150
            params['sigma_lat_ilm'] = 55.56
            params['sigma_lat_isos'] =55.56
            params['sigma_lat_bm'] = 111.13
            params['maxdist'] = 550 # ~100 pixels in spectralis
            params['bsc_indep'] = False
        else:
            print('wrong parameter\n')
        return params
    #detect the retina boundaries and the retina mask will have a value of 0
    #temporarily ignore the argument number checking part

    params = getParams(paramSet) #params is a dictionary here
    

    #maximum distance from ILM to ISOS:
    maxdist = params['maxdist']
    #maximum distance from ISOS to BM:
    maxdist_bm = 116.02
    #Minimum distance from ISOS to BM:
    isosThresh = 20
    #Median filter outlier threshold distance and kernel
    dc_thresh = 10
    mf_k = 140


    #Process B-scans independently
    bsc_indep = params['bsc_indep']
    
    if 'angle' in header:
        if abs(abs(header['angle'])-90) < 25:
            bsc_indep = 1

    #sigma values for smoothing final surfaces
    sigma_tp_ilm = 91.62
    sigma_tp_isos = 91.62
    sigma_tp_bm = 244.32
    #lateral direction
    sigma_lat_ilm = params['sigma_lat_ilm']
    sigma_lat_isos = params['sigma_lat_isos']
    sigma_lat_bm = params['sigma_lat_bm']

    #convert all values frmo micron to pixel
    sz = header['ScaleZ']*1000
    hd = header['Distance']*1000
    sigma_lat = params['sigma_lat']/(header['ScaleX']*1000)
    sigma_ax = params['sigma_ax']/sz
    distConst = np.round(params['distconst']/sz)
    maxdist = np.round(maxdist/sz)
    maxdist_bm = np.round(maxdist_bm/sz)
    isosThresh = np.round(isosThresh/sz)
    dc_thresh = np.round(dc_thresh/sz*(128/6)*header['Distance'])
    
    temp = np.round(np.array([(mf_k/(header['ScaleX']*1000)),(mf_k/(header['Distance']*1000))]))
    mf_k = (temp*2 +1).reshape((1,2))
    sigma_tp_ilm = sigma_tp_ilm/hd
    sigma_tp_isos = sigma_tp_isos/hd
    sigma_tp_bm = sigma_tp_bm/hd
    sigma_lat_ilm = sigma_lat_ilm/(header['ScaleX']*1000)
    sigma_lat_isos = sigma_lat_isos/(header['ScaleX']*1000)
    sigma_lat_bm = sigma_lat_bm/(header['ScaleX']*1000)

    # #handle zero or nan values on the borders
    img_vol[np.isnan(img_vol)] = 0

    # #fill in from the left side:
    inds = np.argmax(img_vol>0, axis = 1) 

   
    #in matlab the y-axis is not automatically deleted, so here the loop needs to change
    for i in range(img_vol.shape[0]): 
        for j in range(img_vol.shape[2]):
            p = inds[i,j]
            if p > 0 and p < i-1:
                if p < img_vol.shape[1] - 2:
                    #avoid using low intensity edge pixels
                    img_vol[i,:(p+1), j] = img_vol[i,(p+2), j]
                else:
                    img_vol[i,:(p-1), j] = img_vol[i,p,j]
    
    #fill in from the right side
    temp_vol = np.fliplr(img_vol > 0) #index of last nonzero value
    inds = np.argmax(temp_vol>0, axis = 1)
    inds = img_vol.shape[1] - inds -1 #use -1 instead of + 1 for numpy
    
    for i in range(img_vol.shape[0]): 
        for j in range(img_vol.shape[2]):
            p = inds[i,j]
            if p < img_vol.shape[1] and img_vol.shape[1] - p < i:
                if p >2:
                    #avoid using low intensity edge pixels
                    img_vol[i, (p-1):, j] = img_vol[i,(p-2), j]
                else:
                    img_vol[i, (p+1):, j] = img_vol[i,p,j]

    #fill in from top:
    mv = np.mean(img_vol)
    
    #same process for inds
    inds = np.argmax(temp_vol>0, axis = 0)
    for i in range(img_vol.shape[1]):
        for j in range(img_vol.shape[2]):
            p = inds[i,j]
            if p > 0:
                if  p < img_vol.shape[0] -2:
                    #avoid using low intensity edge pixels
                    if img_vol[p+2,i,j] < mv:
                        img_vol[:(p+1),i,j] = img_vol[p+2,i,j]
                    else:
                        #cut through the retina so keep a gradient
                        img_vol[:(p+1), i, j]= mv 
                else:
                    img_vol[:(p-1), i, j] = img_vol[p,i,j]
    #fill in from the bottom

    temp_vol = np.flipud(img_vol > 0) #index of last nonzero value
    inds = np.argmax(temp_vol>0, axis = 0)
    inds = img_vol.shape[0] - inds - 1 #use -1 instead of + 1 for numpy
    for i in range(img_vol.shape[1]):
        for j in range(img_vol.shape[2]):
            p = inds[i,j]
            if p < img_vol.shape[0]:
                if p > 2:
                    #avoid using low intensity edge pixels
                    img_vol[(p-1):, i,j] = img_vol[(p-2),i,j]
                else:
                    img_vol[(p+1):,i,j] = img_vol[p,i,j] 
    
    # #Pre-processing
    
    sigma_ax = float(sigma_ax)
    sigma_lat = float(sigma_lat)
    grad = nd.gaussian_filter(img_vol, sigma = (sigma_ax,0, 0), mode='nearest', order=0,truncate=2*np.round(2*sigma_ax) + 1) 
    grad = nd.gaussian_filter(grad, sigma = (0,sigma_lat,0), mode='nearest', order=0,truncate=2*np.round(2*sigma_lat) + 1)
    # for i in range(grad.shape[-1]):
    #     grad[:,:,i] = nd.sobel(grad[:,:,i], mode='nearest', axis =0)
    grad = nd.sobel(grad, mode='nearest', axis =0)
    grad_o = grad.copy()
    max1pos = np.argmax(grad, axis =0)
    
    #to check if max1pos is vector, we have to use the shape of max1pos
    m_size = max1pos.shape
    if m_size[0] == 1 or m_size[1] == 1:
        print('reach here') #shouldn't reach here with given input
        max1pos =np.transpose(max1pos)
 

    #Find the largesr gradient to the max gradient at distance of
    #at least distCount away but not more than maxdist away
    for i in range(grad.shape[1]):
        for j in range(grad.shape[2]):
            dc = distConst
            if (max1pos[i,j] - distConst) < 1:
                dc = max1pos[i,j] -1
            elif (max1pos[i,j] + distConst) > grad.shape[0]:
                dc = grad.shape[0] - max1pos[i,j]

            grad[int(max1pos[i,j]-dc):int(max1pos[i,j]+dc), i,j] = 0
            #max distance
            if (max1pos[i,j] - maxdist) > 0:
                grad[:int(max1pos[i,j]-maxdist),i,j] = 0
            if (max1pos[i,j] + maxdist) <= grad.shape[0]:
                grad[int(max1pos[i,j]+maxdist):,i,j] = 0

   
            
    max2pos = np.argmax(grad, axis =0)
    m2_size  =max2pos.shape 
    if m2_size[0] == 1 or m2_size[1] == 1:
        max2pos =np.transpose(max2pos)
    
    ilm = np.minimum(max1pos, max2pos)
    isos = np.maximum(max1pos, max2pos) 

    #Fill in BM boundary
    grad = grad_o
    
    #BM is largest negative gradient below the ISOS
    for i in range(grad.shape[1]):
        for j in range(grad.shape[2]):
            grad[:int(isos[i,j]+isosThresh), i ,j] = 0
            if (isos[i,j]+maxdist_bm) <= grad.shape[0]:
                grad[int(isos[i,j]+maxdist_bm):,i,j] = 0



    #To encourage boundary points closer to the top of the image, weight linearly by depth
    isos_temp = (grad.shape[0] - (isos[np.newaxis,:,:]  + maxdist_bm))
    lin = np.transpose(np.arange(grad.shape[0])).reshape(496,1,1) + isos_temp
    lin = -0.5/grad.shape[0] * lin +1
    grad = grad*lin
  
    bot = np.argmin(grad, axis = 0) #no need to squeeze for python
    bot_sz  = bot.shape
    if bot_sz[0] == 1 or bot_sz[1] == 1:
        print('reach here') #shouldn't reach here with given input
        bot =np.transpose(bot)
    bm  = bot 



    #detect outliers
    if bsc_indep: #not reached in the given data
        th = bm - ilm
        print(bm.shape)
        print(dc_thresh.shape)
        th_med = sg.medfilt2d(th, mf_k.reshape(1,2))
        bpt = (abs(th - th_med) > dc_thresh)
    else:
        mf_k = mf_k.astype(int)
        ilm_med = nd.median_filter(ilm.astype(float), [mf_k[0,0], mf_k[0,1]])
        isos_med = nd.median_filter(isos.astype(float), [mf_k[0,0], mf_k[0,1]])
        bm_med = nd.median_filter(bm.astype(float), [mf_k[0,0], mf_k[0,1]])
        dc_thresh = float(dc_thresh)
        ilmt = np.abs(ilm - ilm_med)
        isost = np.abs(isos - isos_med)
        bmt = np.abs(bm - bm_med)
        par = np.maximum(ilmt, isost)
        par = np.maximum(par, bmt) #the combined maximum of three absolute difference
        bpt = par > dc_thresh

    #Fill in outlier points:
    ilm = ilm.astype(float)
    isos = isos.astype(float)
    bm = bm.astype(float)
    ilm[bpt] = np.nan  #find correspondance of nan
    isos[bpt] = np.nan
    bm[bpt] = np.nan
    nbpt = 0

    
    
    
    if np.any(np.any(bpt)): #since bpt is 2-D
        nbpt = np.sum(bpt)
        if bsc_indep: #not reached, not fully implemented
            x = np.transpose(np.range(ilm.shape[0]))
            for j in range(ilm.shape[1]):
                #p = polyfit polyfit function
                #bm[:,j] = polyval(p,x) # function need to be transfered
                quit()
            #linearly interpolate ILM and ISOS
            nv = any(np.isnan(ilm))
            xpts = np.arange(ilm.shape[0])
            for j in range(ilm.shape[1]):
                if nv[j]:
                    nv2 = not np.isnan(ilm)
                    #ilm[:,j = interp1 need to transfer interpolate
                    #isos[:,j] = interp1
                    #bm = interp1
        else:
            #temporary replacement of the inpaint_nan function
            ilm = inan.inpaint_nans(ilm)
            isos = inan.inpaint_nans(isos)
            bm = inan.inpaint_nans(bm) 
    
    #Get final boundaries by smoothing
    #smooth surfaces
    sigma_tp_ilm = float(sigma_tp_ilm)
    sigma_tp_isos = float(sigma_tp_isos)
    sigma_tp_bm = float(sigma_tp_bm)
    sigma_lat_ilm = float(sigma_lat_ilm)
    sigma_lat_isos = float(sigma_lat_isos)
    sigma_lat_bm = float(sigma_lat_bm)
    if not bsc_indep:
        ilm = nd.gaussian_filter(ilm, sigma = (sigma_tp_ilm, 0), mode='nearest', order=0, truncate=2*np.round(3*sigma_tp_ilm) + 1)
        isos = nd.gaussian_filter(isos, sigma = (sigma_tp_isos, 0), mode='nearest', order=0,truncate=2*np.round(3*sigma_tp_isos) + 1)
        bm = nd.gaussian_filter(bm, sigma = (sigma_tp_bm, 0), mode='nearest', order=0, truncate=2*np.round(3*sigma_tp_bm) + 1)
        bm = nd.gaussian_filter(bm, sigma = (0, sigma_lat_bm), mode='nearest', order=0, truncate=2*np.round(3*sigma_lat_bm) + 1)
 
    ilm = nd.gaussian_filter(ilm, sigma = (0, sigma_lat_ilm), mode='nearest', order=0, truncate=2*np.round(3*sigma_lat_ilm) + 1)
    isos = nd.gaussian_filter(isos, sigma = (0, sigma_lat_isos), mode='nearest', order=0, truncate=2*np.round(3*sigma_lat_isos) + 1)
    #need to transfer all the image to filter function
    #Enforce ordering and a very small minimum thickness

    bmilm = (bm -ilm)*header['ScaleZ']*1000 <100
    ilm[bmilm] = bm[bmilm] - 100/header['ScaleZ']/1000
    bmisos = (bm -isos)*header['ScaleZ']*1000 <10
    isos[bmisos] = bm[bmisos] - 10/header['ScaleZ']/1000
    isosilm = (isos-ilm)*header['ScaleZ']*1000 < 90
    isos[isosilm] = ilm[isosilm] + 90/header['ScaleZ']/1000

    #Make sure that we are not out of the volumn
    ilm[ilm <1] = 1
    ilm[ilm> img_vol.shape[0]] = img_vol.shape[0]
    isos[isos <1] = 1
    isos[isos > img_vol.shape[0]] = img_vol.shape[0]
    bm[bm<1] = 1
    bm[bm>img_vol.shape[0]] = img_vol.shape[0]
   
    #create mask volume
    retinaMask = np.zeros(img_vol.shape)
    for i in range(img_vol.shape[1]):
        for j in range(grad.shape[2]):
            retinaMask[int(np.round(ilm[i,j])):int(np.round(isos[i,j])), i, j] = 1
            retinaMask[int(np.round(isos[i,j])):int(np.round(bm[i,j])), i, j] =2
    ilm_cat = ilm.reshape(ilm.shape[0], ilm.shape[1], 1)
    isos_cat = isos.reshape(isos.shape[0], isos.shape[1], 1)
    bm_cat = bm.reshape(bm.shape[0], bm.shape[1], 1)
    
    boundaries = np.concatenate((ilm_cat, isos_cat, bm_cat), axis= 2)
    #define the shift amount here
    stemp = np.mean(bm, axis=0) + np.round(img_vol.shape[0]/2) - np.mean(bm, axis=0)
    shifts = bm - stemp.reshape((1,-1))

    # plt.imshow(img_vol[:,:,0])
    # plt.plot(ilm[:,0])
    # plt.plot(isos[:,0])
    # plt.plot(bm[:,0])
    # plt.show()
    # quit()
    

    return [retinaMask, shifts, boundaries, nbpt]
    



