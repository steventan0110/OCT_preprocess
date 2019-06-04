## OCT_preprocess by Weiting Tan
Preprocess of Optical coherence tomography (OCT) includes following steps:
1. <code>octSpectrailisReader</code> convert OCT image into python processable nd-array and retrieve useful information in the image
OCT image in the first layer shown by matlab.pyplot.imshow:

2. <code>retinaDetect</code> find the boundaries of inner limiting membrane(ILM), inner segment(IS), outer segment (OS)
and Bruch’s membrane (BM)
3. <code>normalizeOCT</code> normalize and reduce noise of the OCT image
4. <code>retinaFlatten</code> calculate the shitfs based on return value in <code>retinaDetect</code> and flatten the image using
BM as baseline.

