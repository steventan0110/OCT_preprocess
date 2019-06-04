import numpy as np
def  octSpectralisReader(path, **kwargs):
    r'''
    OPENVOL Read Heidelberg Engineering (HE) OCT raw files (VOL ending)
    [HEADER, BSCANHEADER, SLO, BSCANS] = OPENVOL(PATH, OPTIONS)
    This function performs volume OCT data (xxxx.vol) reading. 
    header:header information as described by HE. Struct with each entry
    named by the HE conventions. 
    BSCANHEADER: B-scanheader information. Struct with each entry named by 
    the HE conventions. The entries consist of vectors/matrices, with one 
    field per B-Scan. 
    SLO: Slo image as unsigned integers.
    BScans: BScans. 3D matrix with floating point data of the B-Scans.
    PATH: Filename of the VOL-file to read, with ending.
    return [header, BScanHeader, slo, BScans]'''
    if 'options' in kwargs.keys():
        options = kwargs['options']
    else:
        options = {}
    fid = open(path)
    header = {}
    header['Version'] = np.fromfile( fid, count=12, dtype=np.int8)
    header['SizeX'] = np.fromfile( fid, count=1, dtype=np.int32)
    header['NumBScans'] = np.fromfile( fid, count=1, dtype=np.int32)
    header['SizeZ'] = np.fromfile( fid, count=1, dtype=np.int32)
    header['ScaleX'] = np.fromfile( fid, count=1, dtype=np.double )
    header['Distance'] = np.fromfile( fid, count=1, dtype=np.double )
    header['ScaleZ'] = np.fromfile( fid, count=1, dtype=np.double )
    header['SizeXSlo'] = np.fromfile( fid, count=1, dtype=np.int32 )
    header['SizeYSlo'] = np.fromfile( fid, count=1, dtype=np.int32 )
    header['ScaleXSlo'] = np.fromfile( fid, count=1, dtype=np.double )
    header['ScaleYSlo'] = np.fromfile( fid, count=1, dtype=np.double )
    header['FieldSizeSlo'] = np.fromfile( fid, count=1, dtype=np.int32)
    header['ScanFocus'] = np.fromfile( fid, count=1, dtype=np.double )
    header['ScanPosition'] = ''.join(map(chr,np.fromfile( fid, count=4, dtype=np.uint8)))
    ### the exam time has 1 bit difference with matlab readin value. dont know why
    header['ExamTime'] = np.fromfile( fid, count=1, dtype=np.int64 )
    header['ScanPattern'] = np.fromfile( fid, count=1, dtype=np.int32 )
    header['BScanHdrSize'] = np.fromfile( fid, count=1, dtype=np.int32 ) 
    header['ID'] = ''.join(map(chr,np.fromfile( fid, count=16, dtype=np.uint8))) 
    header['ReferenceID'] = ''.join(map(chr,np.fromfile( fid, count=16, dtype=np.uint8))) 
    header['PID'] = np.fromfile( fid, count=1, dtype=np.int32 ) 
    header['PatientID'] = ''.join(map(chr,np.fromfile( fid, count=21, dtype=np.uint8))) 
    header['Padding'] = np.fromfile( fid, count=3, dtype=np.int8 ) 
    header['DOB'] = np.fromfile( fid, count=1, dtype=np.double ) 
    header['VID'] = np.fromfile( fid, count=1, dtype=np.int32 ) 
    header['VisitID'] = ''.join(map(chr,np.fromfile( fid, count=24, dtype=np.uint8))) 
    header['VisitDate'] = np.fromfile( fid, count=1, dtype=np.double ) 
    header['GridType'] = np.fromfile( fid, count=1, dtype=np.int32) 
    header['GridOffset'] = np.fromfile( fid, count=1, dtype=np.int32) 
    header['Spare'] = np.fromfile( fid, count=1832, dtype=np.int8 ) 
    if hasattr(options,'header'):
        return [header]

    # read slo
    fid.seek(2048,0)
    slo = np.fromfile(fid, count= \
                    int(header['SizeXSlo']*header['SizeYSlo']),\
                    dtype=np.uint8)
    slo = slo.reshape((int(header['SizeXSlo']),int(header['SizeYSlo'])))

    # read Bscans
    BScans=np.zeros((header['SizeZ'][0], header['SizeX'][0] ,header['NumBScans'][0]), dtype=np.float32)
    BScanHeader = {}
    BScanHeader['StartX'] = np.zeros(((header['NumBScans'][0],)),dtype=np.float64)
    BScanHeader['StartY'] = np.zeros(((header['NumBScans'][0],)), dtype=np.float64) 
    BScanHeader['EndX'] = np.zeros((header['NumBScans'][0],),dtype= np.float64) 
    BScanHeader['EndY'] = np.zeros((header['NumBScans'][0],), dtype=np.float64) 
    BScanHeader['NumSeg'] = np.zeros((header['NumBScans'][0],), dtype=np.int32) 
    BScanHeader['Quality'] = np.zeros((header['NumBScans'][0],), dtype=np.float32) 
    BScanHeader['Shift'] = np.zeros((header['NumBScans'][0],),dtype= np.int32) 
    BScanHeader['ILM'] = np.zeros((header['NumBScans'][0],header['SizeX'][0]), dtype=np.float32) 
    BScanHeader['RPE'] = np.zeros((header['NumBScans'][0],header['SizeX'][0]), dtype=np.float32) 
    BScanHeader['NFL'] = np.zeros((header['NumBScans'][0],header['SizeX'][0]), dtype=np.float32) 

    for zz in range(header['NumBScans'][0]):
        fid.seek(int(16+2048+(header['SizeXSlo']*header['SizeYSlo'])+\
        (zz*(header['BScanHdrSize']+header['SizeX']*header['SizeZ']*4))), 0)
        StartX = np.fromfile(fid, count=1, dtype=np.float64)
        StartY = np.fromfile(fid, count=1, dtype=np.float64)
        EndX = np.fromfile(fid, count=1, dtype=np.float64)
        EndY = np.fromfile(fid, count=1, dtype=np.float64)
        NumSeg = np.fromfile(fid, count=1, dtype=np.int32)
        OffSeg = np.fromfile(fid, count=1, dtype=np.int32)
        Quality = np.fromfile(fid, count=1, dtype=np.float32)
        Shift = np.fromfile(fid, count=1, dtype=np.int32)

        BScanHeader['StartX'][zz] = StartX
        BScanHeader['StartY'][zz] = StartY
        BScanHeader['EndX'][zz] = EndX
        BScanHeader['EndY'][zz] = EndY
        BScanHeader['NumSeg'][zz] = NumSeg
        BScanHeader['Quality'][zz] = Quality
        BScanHeader['Shift'][zz] = Shift

        fid.seek(header['BScanHdrSize']+2048+(header['SizeXSlo']*header['SizeYSlo']) \
                +(zz*(header['BScanHdrSize']+header['SizeX']*header['SizeZ']*4)), 0)
        octs = np.fromfile(fid, count=int(header['SizeX']*header['SizeZ']), dtype=np.float32)
        octs = np.sqrt(np.sqrt(octs))
        octs[octs>1] = 0
        BScans[:,:,zz] = octs.reshape((header['SizeX'][0],header['SizeZ'][0]),order='F').transpose()
    return [header, BScanHeader, slo, BScans]

