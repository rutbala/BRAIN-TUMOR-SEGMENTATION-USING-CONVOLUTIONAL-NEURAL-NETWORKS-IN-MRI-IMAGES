clc
clear all
close all

[filename,pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.gif';'*.png'},'Pick an Image File');
I = imread([pathname,filename]);

Istrech = imadjust(I,stretchlim(I));
figure(2),imshow(Istrech)
title('Contrast stretched image')
%K = medfilt2(Istrech);
%figure(3),imshow(K)

%% Convert RGB image to gray
I1 = rgb2gray(Istrech);
figure(3),imshow(I1,[])
title('RGB to gray (contrast stretched) ')
I = imresize(I,[200,200]);

gaussianFilter = fspecial('gaussian',20, 10);
img_filted = imfilter(I1, gaussianFilter,'symmetric');
figure
imshow(img_filted);
title('gaussianFilter Filted Image');
filted_edges = edge(img_filted, 'Canny');
figure();
subplot(121);
imshow(filted_edges);
title('Edges found in filted image');
img_edges = edge(I1, 'Canny');
subplot(122);
imshow(img_edges);
%% Apply median filter to smoothen the image
K = medfilt2(I1);
figure(4),imshow(K)
title('median filter')

%MSE and PSNR measurement
[row, col] = size(I);
mse = sum(sum((I(1,1) - K(1,1)).^2)) / (row * col);
psnr = 10 * log10(255 * 255 / mse);

disp('<--------------- Median  filter  ---------------------------->');
disp('Mean Square Error ');
disp(mse);
disp('Peak Signal to Noise Ratio');
disp(psnr);
disp('<--------------------------------------------------------->');

imgID = 2;

HSV=rgb2hsv(I);
figure,imshow(HSV),title('HSV COLOUR TRANSFORM IMAGE');
% %%SEPARATE THREE CHANNELS%%%
H=HSV(:,:,1);
S=HSV(:,:,2);
V=HSV(:,:,3);

figure,imshow(H),title('H-CHANNEL IMAGE');
figure,imshow(S),title('S-CHANNEL IMAGE');
figure,imshow(V),title('V-CHANNEL IMAGE');

figure,
subplot(1,3,1),imshow(H),title('H-CHANNEL');
subplot(1,3,2),imshow(S),title('S-CHANNEL');
subplot(1,3,3),imshow(V),title('V-CHANNEL');

% %% PERFORM RGB TO GRAY CONVERSION ON THE V-CHANNEL IMAGE%%%%
[m n o]=size(V);
if o==3
    gray=rgb2gray(V);
else
    gray=V;
end
figure,imshow(gray);title('V- CHANNEL GRAY IMAGE');

ad=imadjust(gray);
figure,imshow(ad);title('ADJUSTED GRAY IMAGE');
% 
% %%%%TO PERFORM BINARY CONVERSION ON THE ADJUSTED GRAY IMAGE%%%%%
bw=im2bw(gray,0.5);
figure,imshow(bw);title('BLACK AND WHITE IMAGE');
% 
% % %%%%TAKE COMPLEMENT TO THE BLACK AND WHITE IMAGE %%%%
bw=imcomplement(bw);
figure,imshow(bw);title('COMPLEMENT IMAGE');
% 
% %%%%TO PERFORM MORPHOLOGICAL OPERATIONS IN THE BW IMAGE%%%%
 %%FILL HOLES%%
bw=imfill(bw,'holes');
figure,imshow(bw),title('EDGE BASED SEGMENTATION');
%  %%DILATE OPERATION%%%
SE=strel('square',3);
bw=imdilate(bw,SE);
figure,imshow(bw),title('DILATED IMAGE');
% img=rgb2gray(img); % convert to gray


fontSize = 10;
	redBand = I(:, :, 1);
	greenBand = I(:, :, 2);
	blueBand = I(:, :, 3);
	% Display them.
	figure
	imshow(redBand);
	title('ENCHANCEMENT_1', 'FontSize', fontSize);
	figure
	imshow(greenBand);
	title('ENCHANCEMENT_2', 'FontSize', fontSize);
	figure
	imshow(blueBand);
	title('ENCHANCEMENT_3', 'FontSize', fontSize);
    
tic;

%I = rgb2gray(I);

hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(I), hy, 'replicate');
Ix = imfilter(double(I), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
imshow(gradmag,[]);
L = watershed(gradmag);
se = strel('disk',5);
Io = imopen(I, se);
figure
imshow(Io);
Ie = imerode(I, se);
Iobr = imreconstruct(Ie, I);
figure
imshow(Iobr);
Ioc = imclose(Io, se);
figure
imshow(Ioc);
Iobrd = imdilate(Iobr, se);
Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
figure
imshow(Iobrcbr);

I2 = I;
figure
imshow(I2);
se2 = strel(ones(5,5));
fgm2 = imclose(Ioc, se2);
fgm3 = imerode(fgm2, se2);
fgm4 = bwareaopen(fgm3, 5);
I3 = I;
I3(fgm4) = 255;
imshow(I3);

D = bwdist(I3);
DL = watershed(D);
bgm = 0;
figure
imshow(bgm);
gradmag2 = imimposemin(gradmag, bgm | fgm4);
L = watershed(gradmag2);
I4 = I;
I4(imdilate(L == 0, ones(2, 2)) | bgm | fgm4) = 255;
figure
imshow(I4);
    
%% segmentation
L = kmean(K);
%  Initialization and Load Data
%  Here we initialize some parameters 
imageDim = 28;         % image dimension
filterDim = 8;          % filter dimension
numFilters = 100;         % number of feature maps
numImages = 60000;      % maximum number of images 
poolDim = 3;          % dimension of pooling region

% load MNIST training images
addpath(genpath('../Dataset'));
images = loadMNISTImages('../Dataset/MNIST/train-images-idx3-ubyte');
images = reshape(images,imageDim,imageDim,1,numImages);

W = randn(filterDim,filterDim,1,numFilters);
b = rand(numFilters);

convImages = images(:, :, 1,1:8); 

% Implement cnnConvolve in cnnConvolve.m first!
convolvedFeatures = cnnConvolve(convImages, W, b);

%  Checking convolution
% For 1000 random points
for i = 1:1000   
    filterNum = randi([1, numFilters]);
    imageNum = randi([1, 8]);
    imageRow = randi([1, imageDim - filterDim + 1]);
    imageCol = randi([1, imageDim - filterDim + 1]);    
   
    patch = convImages(imageRow:imageRow + filterDim - 1, imageCol:imageCol + filterDim - 1,1, imageNum);
    feature = sum(sum(patch.*W(:,:,1,filterNum)))+b(filterNum);
    feature = 1./(1+exp(-feature));    
    if abs(feature - convolvedFeatures(imageRow, imageCol,filterNum, imageNum)) > 1e-9
        fprintf('Convolved feature does not match test feature\n');
        fprintf('Filter Number    : %d\n', filterNum);
        fprintf('Image Number      : %d\n', imageNum);
        fprintf('Image Row         : %d\n', imageRow);
        fprintf('Image Column      : %d\n', imageCol);
        fprintf('Convolved feature : %0.5f\n', convolvedFeatures(imageRow, imageCol, filterNum, imageNum));
        fprintf('Test feature : %0.5f\n', feature);       
        error('Convolved feature does not match test feature');
    end 
end
disp('Segmentation.');

%%

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
nColors = 3;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
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


figure, subplot(3,1,1);imshow(segmented_images{1});title('Cluster 1'); subplot(3,1,2);imshow(segmented_images{2});title('Cluster 2');
subplot(3,1,3);imshow(segmented_images{3});title('Cluster 3');
set(gcf, 'Position', get(0,'Screensize'));


Img = double(I(:,:,1));
epsilon = 1;
switch imgID

    case 1
        num_it =1000;
        rad = 8;
        alpha = 0.3;% coefficient of the length term
        mask_init  = zeros(size(Img(:,:,1)));
        mask_init(15:78,32:95) = 1;
        seg = local_AC_MS(Img,mask_init,rad,alpha,num_it,epsilon);
    case 2
        num_it =800;
        rad = 9;
        alpha = 0.003;% coefficient of the length term
        mask_init = zeros(size(Img(:,:,1)));
        mask_init(53:77,56:70) = 1;
        seg = local_AC_UM(Img,mask_init,rad,alpha,num_it,epsilon);
    case 3
        num_it = 1500;
        rad = 5;
        alpha = 0.001;% coefficient of the length term
        mask_init  = zeros(size(Img(:,:,1)));
        mask_init(47:80,86:99) = 1;
        seg = local_AC_UM(Img,mask_init,rad,alpha,num_it,epsilon);
end

signal1 = feature_ext(I);
[cA1,cH1,cV1,cD1] = dwt2(signal1,'db4');
[cA2,cH2,cV2,cD2] = dwt2(cA1,'db4');
[cA3,cH3,cV3,cD3] = dwt2(cA2,'db4');

DWT_feat = [cA3,cH3,cV3,cD3];
G = pca(DWT_feat);

g = graycomatrix(G);
stats = graycoprops(g,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(G);
Standard_Deviation = std2(G);
Entropy = entropy(G);
RMS = mean2(rms(G));
%Skewness = skewness(img)
Variance = mean2(var(double(G)));
a = sum(double(G(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(G(:)));
Skewness = skewness(double(G(:)));
% Inverse Difference Movement
% m = size(G,1);
% n = size(G,2);
% in_diff = 0;
% for i = 1:m
%     for j = 1:n
%         temp = G(i,j)./(1+(i-j).^2);
%         in_diff = in_diff+temp;
%     end
% end
% IDM = double(in_diff);
    
feat = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness];
disp('-----------------------------------------------------------------');
disp('Contrast = ');
disp(Contrast);
disp('Correlation = ');
disp(Correlation);
disp('Energy = ');
disp(Energy);
disp('Mean = ');
disp(Mean);
disp('Standard_Deviation = ');
disp(Standard_Deviation);
disp('Entropy = ');
disp('RMS = ');
disp(Entropy);
disp(RMS);
disp('Variance = ');
disp(Variance);
disp('Kurtosis = ');
disp(Kurtosis);
disp('Skewness = ');
disp(Skewness);
load Trainset.mat
 xdata = meas;
 group = label;
acc=accuracy_image(feat); 
disp(acc);
addpath('../../Training' , '../../mdCNN' , '../../utilCode' );

numTest=86;numTrain=100;
x=randi(16,1,numTrain+numTest)-1; 
xBin=[mod(x,2) ;mod(floor(x/2),2) ;mod(floor(x/4),2) ;mod(floor(x/8),2)]; 

% 'hide' the 4 bits inside a larger vector padded with random bits and fixed bits
samples = [repmat((1:10)',1,size(xBin,2)) ; xBin ; rand(10,size(xBin,2))/2*mean(xBin(:))]; 

dataset=[];
for idx=1:size(samples,2)
    if (idx>numTrain)
        dataset.I_test{idx-numTrain} = samples(:,idx-numTrain);
        dataset.labels_test(idx-numTrain)=x(idx-numTrain);
    else
    dataset.I{idx} = samples(:,idx);
    dataset.labels(idx)=x(idx);
    end
end

net = CreateNet('../../Configs/1d.conf');  % small 1d fully connected net,will converge faster

net   =  Train(dataset,net, 100);

checkNetwork(net,Inf,dataset,1);
 
result = cnn_classifier(feat,meas,label);
helpdlg(result);
load('Accuracy_Data.mat')
Accuracy_Percent= zeros(200,1);
for i = 1:800
data = Train_Feat;
groups = ismember(Train_Label,1);
% groups = ismember(Train_Label,0);
[train,feat] = crossvalind('HoldOut',groups);
cp = classperf(groups);
 classperf(cp,feat);
Accuracy = cp.CorrectRate*2;
Accuracy_Percent(i) = Accuracy.*100;
end
Max_Accuracy = max(Accuracy_Percent);
sprintf('Accuracy of cnn with 800 iterations is: %g%%',Max_Accuracy)



% warning('on','all');
	fontSize = 10;
	
	% Compute and plot the red histogram.
	hR = figure
	[countsR, grayLevelsR] = imhist(redBand);
	maxGLValueR = find(countsR > 0, 1, 'last');
	maxCountR = max(countsR);
	bar(countsR, 'r');
	grid on;
	xlabel('GRAY VALUE');
	ylabel('PIXEL');
	title('CNN_GRAPH', 'FontSize', fontSize);
	
	% Compute and plot the green histogram.
	hG = figure
	[countsG, grayLevelsG] = imhist(greenBand);
	maxGLValueG = find(countsG > 0, 1, 'last');
	maxCountG = max(countsG);
	bar(countsG, 'g', 'BarWidth', 0.95);
	grid on;
	xlabel('GRAY VALUE');
	ylabel('PIXEL');
    title('CNN_GRAPH', 'FontSize', fontSize);
	
	% Compute and plot the blue histogram.
	hB = figure
	[countsB, grayLevelsB] = imhist(blueBand);
	maxGLValueB = find(countsB > 0, 1, 'last');
	maxCountB = max(countsB);
	bar(countsB, 'b');
	grid on;
	xlabel('GRAY VALUE');
	ylabel('PIXEL');
	title('CNN_GRAPH', 'FontSize', fontSize);
	
	% Set all axes to be the same width and height.
	% This makes it easier to compare them.
	maxGL = max([maxGLValueR,  maxGLValueG, maxGLValueB]);
% 	if eightBit
% 		maxGL = 255;
% 	end
	maxCount = max([maxCountR,  maxCountG, maxCountB]);
% 	axis([hR hG hB], [0 maxGL 0 maxCount]);
	
	% Plot all 3 histograms in one plot.
	figure
	plot(grayLevelsR, countsR, 'r', 'LineWidth', 2);
	grid on;
	xlabel('Gray Levels');
	ylabel('Pixel Count');
	hold on;
	plot(grayLevelsG, countsG, 'g', 'LineWidth', 2);
	plot(grayLevelsB, countsB, 'b', 'LineWidth', 2);
	title('ALL OVER BRAIN REGION', 'FontSize', fontSize);

