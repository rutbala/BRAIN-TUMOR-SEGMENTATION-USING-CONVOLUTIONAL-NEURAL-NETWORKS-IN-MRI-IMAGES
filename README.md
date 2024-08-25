# BRAIN-TUMOR-SEGMENTATION |  Image Processing, Machine Learning, MATLAB

Introduction:-

A tumor is an abnormal growth caused by uncontrolled cell division, with brain tumors being particularly dangerous. Brain tumors are classified into four grades, and their treatment depends on the tumor's type, grade, and location. Early detection is critical as these tumors can be fatal. MRI is commonly used for diagnosis, but manual segmentation is impractical due to the large data volume. We propose an automatic segmentation method using Convolutional Neural Networks (CNN) with small 3×3 kernels, enabling a deeper architecture and reducing overfitting. Data augmentation further enhances the accuracy of brain tumor segmentation in MRI images.

Prposed System:- 

In this project, MRI images are used to detect brain tumors using a D-planning technique involving preprocessing, segmentation, and classification. Preprocessing enhances images by adjusting brightness, contrast, and removing noise. Segmentation methods like K-means, ACM, and Watershed are applied to highlight affected areas. The image is then filtered using a median filter and segmented using Otsu’s thresholding. Features are extracted using Haar wavelet transform (HWT) and Histogram of Oriented Gradients (HOG). Finally, a CNN classifies the extracted features against a trained dataset to determine if a tumor is present.

The software used is MATLAB

![image](https://github.com/rutbala/BRAIN-TUMOR-SEGMENTATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS-IN-MRI-IMAGES/assets/165860969/4e3befe7-bf0a-4b00-a0ce-5328a45ba4cf)

![Screenshot (186)](https://github.com/rutbala/BRAIN-TUMOR-SEGMENTATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS-IN-MRI-IMAGES/assets/165860969/4f4c67e0-03fa-46ed-9507-762328f4927a)

MRI Input Images ( present in the files above)

Result:-

![Screenshot (187)](https://github.com/rutbala/BRAIN-TUMOR-SEGMENTATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS-IN-MRI-IMAGES/assets/165860969/33229e00-fa00-4c8b-a698-64d0f8d8d605)

![Screenshot (188)](https://github.com/rutbala/BRAIN-TUMOR-SEGMENTATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS-IN-MRI-IMAGES/assets/165860969/8b96f0d8-0d86-473b-875d-c1e20b486cf6)

![Screenshot (189)](https://github.com/rutbala/BRAIN-TUMOR-SEGMENTATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS-IN-MRI-IMAGES/assets/165860969/dd947834-010a-429e-9ca5-35dce228dc93)

![Screenshot (190)](https://github.com/rutbala/BRAIN-TUMOR-SEGMENTATION-USING-CONVOLUTIONAL-NEURAL-NETWORKS-IN-MRI-IMAGES/assets/165860969/54858918-48bf-4425-b5d4-398e9cd33ffd)





