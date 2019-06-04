#function used by preprocess in the Normalize and data flatten stage
#written by Weiting Tan on May 15th 2019
import numpy as np
import normalizeOCTVolumn as nov
import retinaDetect as rd 
import retinaFlatten as rf
import matplotlib.pyplot as plt
import cv2
#pp_params is a dictionary that has five fields
def preprocessData(*argv):
    #generate retina mask
    # print(fids,'Normalizing intensities...')
    # try:
    #     if pp_params['normalize'] ==1:
    #         #median filter contrast stretching normalization
    #         img_vol = nov(img_vol, 2, header)
    #     elif pp_params['normalize'] == 2:
    #         #attenuation correction
    #         img_vol = nov(img_vol, 5, header)
    #     else:
    #         img_vol = nov(img_vol, pp_params['normalize'], header)
    # except:
    #     print('error thrown')
    #     img_vol = []
    #     return
 
    img_vol = argv[0]
    header = argv[1]
    pp_params = argv[2]
    scanner_type = argv[3]
    #fids = argv[4] unsued since the program is not complete


    if len(argv) < 6:
        seg_pts = []
    else: seg_pts = argv[5]
    # cv2.imshow('before process', img_vol[:,:,0])
    # cv2.waitKey(0)
    print('Detecting retina boundaries...')
    #print(pp_params['fast_rpe']) for test purpose 
    
    try:
        if pp_params['fast_rpe'] is True:
            if scanner_type == 'cirrus':
                img_vol = img_vol.astype(float)
                # quickFindRPE not finished, but this block is not executed so it's fine
        else:
            if scanner_type == 'spectralis':
                temp_img = np.copy(img_vol)
                [retina_mask, shifts, bds, nbpt] = rd.retinaDetector(temp_img,header,pp_params['retinadetector_type'],False)   
                
            else:
                #median filter
                sz = img_vol.shape
                if  sz.shape[1] == 2:
                    sz[2] = 1
                
                dn_k = [3,3,1]
                #not sure about the meaning of following steps
                # img_vol_mf = permute(img_vol,[2 1 3]);
                # img_vol_mf = medfilt2(img_vol_mf(:,:),[dn_k(2) dn_k(1)],'symmetric');
                # img_vol_mf = reshape(img_vol_mf,sz(2),sz(1),sz(3));
                # img_vol_mf = permute(img_vol_mf,[2 1 3]);

                # img_vol_mf = im2double(img_vol_mf)
                # img_vol = im2double img_vol)
                [retina_mask, shifts, bds, nbpt] = rd.retinaDetector(img_vol_mf,header,pp_params.retinadetector_type,False); 
                
        print('done! '+str(nbpt)+' outlier points\n')
        
        retina_mask =retina_mask > 0
        if nbpt > 0.5*img_vol.shape[1]:
            print('poor fit of retina boundaries detected. Check for artifacts in the data.\n')
        
    except Exception as e:
        print('Error' + str(e))
        img_vol = []
        quit()

    
    if seg_pts: #if there' input for seg_pts, not reached
        #manual segmentation
        #I assume the seg_pts are not already numpy array
        seg = np.array(seg_pts)
        rpe = seg[:,:,1]
        isos = seg[:,:,7]
        bm = seg[:,:,9]

        bds_seg = np.concatenate((rpe, isos, bm),axis =2)
        bds_r = np.round(bds_seg)
        if scanner_type == 'cirrus':
            img_vol = img_vol.astype(float)
            return
        retina_mask = np.zeros(img_vol.shape, dtype = bool)
        for i in range(img_vol.shape[1]):
            for j in range(img_vol.shape[2]):
                if bds_r[i,j,1] > 0:
                    retina_mask[bds[i,j,1]:bds_r[i,j,3], i, j] =True
        #shifts = bsxfun need to find alternative for binary singleton expansion function
        #I assume the numpy broadcasting already takes care of that
        shifts = bm - (np.mean(bm, 1)+np.round(img_vol.shape[0]/2) - np.mean(bm,1))
    

    if pp_params['normalize'] > 0:
        print('Normalizing intensities')
        try:
            if pp_params['normalize'] == 1:
                img_vol = nov.normalizeOCTVolumn(img_vol, 2, header)
            elif pp_params['normalize'] == 2: #the example case                
                img_vol = nov.normalizeOCTVolumn(img_vol, 4, header)
                # plt.imshow(img_vol[:,:,0], cmap='gray')
                # plt.show()
                # quit()
                  
            else:
                bds_n = bds[:,:,[1,3]]
                img_vol = nov.normalizeOCTVolumn(img_vol, pp_params['normalize'], header, bds_n)
               
        except Exception as e:
            print('Errors occur '+ str(e))
            img_vol =[]
            quit()
    
    
    #Flatten to bottom boundary
    if pp_params['flatten'] == True:
        if ('flatten_to_isos' in pp_params) and (pp_params['flatten_to_isos'] is True):
            if not seg_pts:
                isos = bds[:,:,2]
            else:
                isos = bds_seg[:,:,2]
            #same shift = bsxfun thing need to figure out
            shifts = isos - (np.mean(isos, 1)+np.round(img_vol.shape[0]/2) - np.mean(isos,1))

        tb = bds[:,:,0] - shifts
        if np.any(tb <0): #For the example case, won't get in
            shifts = shifts + np.amin(tb)
            #center
            d = np.amin(img_vol.shape[0] - (bds[:,:,-1] - shifts))
            shifts = shifts - np.amin(d)/2
        print('Flattening data')
       
        try:
            img_vol = rf.retinaFlatten(img_vol, shifts, 'linear')
            retina_mask = rf.retinaFlatten(retina_mask, shifts, 'nearest')
            print('done!\n')
        except Exception as e:
            print(str(e))
            img_vol = []
            quit()

  
    return [img_vol, retina_mask, bds, shifts]



        