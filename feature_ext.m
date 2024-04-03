function signal1 = feature_ext(I)
 % Convert to grayscale
% gray = rgb2gray(I);

% Otsu Binarization for segmentation
level = graythresh(I);
img = im2bw(I,level);
%figure, imshow(img);title('Otsu Thresholded Image');

% K means Clustering to segment tumor

cform = makecform('srgb2lab'); 
% Apply the colorform
lab_he = applycform(I,cform);

% Classify the colors in a*b* colorspace using K means clustering.
% Since the image has 3 colors create 3 clusters.
% Measure the distance using Euclidean Distance Metric.
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 1;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',1);
%[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
% Label every pixel in tha image using results from K means
pixel_labels = reshape(cluster_idx,nrows,ncols);
%figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');

% Create a blank cell array to store the results of clustering
segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end

%
%figure, imshow(segmented_images{1});title('Objects in Cluster 1');

%figure, imshow(segmented_images{2});title('Objects in Cluster 2');

seg_img = im2bw(segmented_images{1});
%figure, imshow(seg_img);title('Segmented Tumor');
%seg_img = img;
% Extract features using DWT
x = double(seg_img);
m = size(seg_img,1);
n = size(seg_img,2);
%signal1 = (rand(m,1));
%winsize = floor(size(x,1));
%winsize = int32(floor(size(x)));
%wininc = int32(10);
%J = int32(floor(log(size(x,1))/log(2)));
%Features = getmswpfeat(signal,winsize,wininc,J,'matlab');

%m = size(img,1);
%signal = rand(m,1);
signal1 = seg_img(:,:);
%Feat = getmswpfeat(signal,winsize,wininc,J,'matlab');
%Features = getmswpfeat(signal,winsize,wininc,J,'matlab');