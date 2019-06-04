## OCT_preprocess by Weiting Tan
Preprocess of Optical coherence tomography (OCT) includes following steps:
1. <code>octSpectrailisReader</code> convert OCT image into python processable nd-array and retrieve useful information in the image
OCT image in the first layer shown by matlab.pyplot.imshow:

![sc1](https://user-images.githubusercontent.com/43892072/58902191-9e613100-86d0-11e9-9951-4d41216586b4.png)
![sc2](https://user-images.githubusercontent.com/43892072/58902221-b33dc480-86d0-11e9-99ce-b78633f5400f.png)

2. <code>retinaDetect</code> find the boundaries of inner limiting membrane(ILM), inner segment(IS), outer segment (OS)
and Bruch’s membrane (BM)
Three lines on the image shown are the ILM, ISOS(combinatino of IS and OS), and BM boundaries detected by the code:

![sc3](https://user-images.githubusercontent.com/43892072/58902329-e08a7280-86d0-11e9-99fc-f4ec1f27bd7b.png)
![sc4](https://user-images.githubusercontent.com/43892072/58902349-eb450780-86d0-11e9-9457-86a5227ba369.png)

3. <code>normalizeOCT</code> normalize and reduce noise of the OCT image
after normalizing the image, grayscale image looks like:
![sc5](https://user-images.githubusercontent.com/43892072/58902386-0152c800-86d1-11e9-81b0-e6aa9e276fa0.png)

4. <code>retinaFlatten</code> calculate the shitfs based on return value in <code>retinaDetect</code> and flatten the image using
BM as baseline.
The final image in both grayscale and RGB:

![sc6](https://user-images.githubusercontent.com/43892072/58902481-2ba48580-86d1-11e9-9042-aa41c5dcda2d.png)
![sc7](https://user-images.githubusercontent.com/43892072/58902490-30693980-86d1-11e9-90ea-b35668c9c46b.png)

