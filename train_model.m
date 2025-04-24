% train_model.m
% This script trains a CNN model for pancreatic cancer detection from CT images.

clc;
clear;
close all;

dataDir = fullfile(pwd, 'training images'); % Path to your data folder

imds = imageDatastore(dataDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'FileExtensions', {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.zip'});
disp(countEachLabel(imds));

[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');

inputSize = [224 224 3];
augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augImdsValidation = augmentedImageDatastore(inputSize, imdsValidation, 'ColorPreprocessing', 'gray2rgb');

numClasses = numel(categories(imdsTrain.Labels));

layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    convolution2dLayer(3, 8, 'Padding', 'same', 'Name', 'conv_1')
    batchNormalizationLayer('Name', 'bn_1')
    reluLayer('Name', 'relu_1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_1')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv_2')
    batchNormalizationLayer('Name', 'bn_2')
    reluLayer('Name', 'relu_2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_2')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_3')
    batchNormalizationLayer('Name', 'bn_3')
    reluLayer('Name', 'relu_3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool_3')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')];

options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augImdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

try
    net = trainNetwork(augImdsTrain, layers, options);
    disp('Network training completed!');
    save('trainedPancreaticCancerNet.mat', 'net');
    disp('Network saved as trainedPancreaticCancerNet.mat.');
catch ME
    disp('Error during network training:');
    disp(ME.message);
end
