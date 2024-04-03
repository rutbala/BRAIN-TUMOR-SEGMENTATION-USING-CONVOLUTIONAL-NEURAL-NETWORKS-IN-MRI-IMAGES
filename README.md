# BRAIN-TUMOR-SEGMENTATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS-IN-MRI-IMAGES

Introduction:-

A tumor is a swelling or abnormal growth resulting from the division of cells in an uncontrolled and disorderly manner. Brain tumors are an exceptionally threatening kind of tumor. There exist several types of brain tumors which are classified into four grades. The process for the medical treatment of brain tumors depends on the type, the grade as well as the location of the tumor. If not detected at the early stages, brain tumors can turn out to be fatal. Magnetic Resonance Imaging (MRI) is a widely used imaging technique to assess these tumors, but the large amount of data produced by MRI prevents manual segmentation in a reasonable time, limiting the use of precise quantitative measurements in the clinical practice. We propose an automatic segmentation method based on Convolutional Neural Networks (CNN), exploring small 3×3 kernels. The use of small kernels allows designing a deeper architecture, besides having a positive effect against overfitting, given the fewer number of weights in the network. Data augmentation to be very effective for brain tumor segmentation in MRI images.

![image](https://github.com/rutbala/BRAIN-TUMOR-SEGMENTATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS-IN-MRI-IMAGES/assets/165860969/4e3befe7-bf0a-4b00-a0ce-5328a45ba4cf)

Prposed System:- 
In this project, MRI Images are used to detect whether the patient has Brain Tumor or not. We use D- planning technique, which uses three processes, they are preprocessing, segmentation, and classification. Classification is a backhand process, that is we train a database using a CRM. Preprocessing enhances the images, by increasing brightness, and contrast, and if there is any noise it removes it. Segmentation is of three types Kmean, ACM, and Watershed. Kmean shows images in black and white, ACM shows the region which is affected and Watershed the images in a blur. Using the mat lab we first run the code, then we need to select the MRI Image that needs to be identified. After this, the selected image may have some noise so the image is filtered using a median filter. Then the shape of the brain will be segmented using Otsu’s thresholding method. Features of the selected image will be extracted using haar wavelet transform (HWT) and histogram of oriented gradients (HOG). The extracted features will be classified with the training dataset using CNN. Finally, the result will be displayed whether the brain is affected or not.

![Screenshot (186)](https://github.com/rutbala/BRAIN-TUMOR-SEGMENTATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS-IN-MRI-IMAGES/assets/165860969/4f4c67e0-03fa-46ed-9507-762328f4927a)

MRI Input Images ( present in the files above)

Software:-

Mat lab 2016(A)- MATLAB is a scientific programming language and provides strong mathematical and numerical support for the implementation of advanced algorithms. It is for this reason that MATLAB is widely used by the image processing and computer vision community. New algorithms are very likely to be implemented first in MATLAB, indeed they may only be available in MATLAB.

Result:-

![Screenshot (187)](https://github.com/rutbala/BRAIN-TUMOR-SEGMENTATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS-IN-MRI-IMAGES/assets/165860969/33229e00-fa00-4c8b-a698-64d0f8d8d605)

![Screenshot (188)](https://github.com/rutbala/BRAIN-TUMOR-SEGMENTATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS-IN-MRI-IMAGES/assets/165860969/8b96f0d8-0d86-473b-875d-c1e20b486cf6)

![Screenshot (189)](https://github.com/rutbala/BRAIN-TUMOR-SEGMENTATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS-IN-MRI-IMAGES/assets/165860969/dd947834-010a-429e-9ca5-35dce228dc93)

![Screenshot (190)](https://github.com/rutbala/BRAIN-TUMOR-SEGMENTATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS-IN-MRI-IMAGES/assets/165860969/54858918-48bf-4425-b5d4-398e9cd33ffd)





