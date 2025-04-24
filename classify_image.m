% classify_image.m
% This script classifies a new CT image using the trained CNN model.

load('trainedPancreaticCancerNet.mat', 'net');

[newImageName, newImagePath] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files (*.jpg, *.jpeg, *.png, *.bmp)'}, 'Select a CT Image for Classification');
if isequal(newImageName, 0)
    disp('User canceled the file selection');
    return;
end

newImage = imread(fullfile(newImagePath, newImageName));
if size(newImage, 3) == 1
    newImage = repmat(newImage, [1, 1, 3]);
end

newImageResized = imresize(newImage, [224, 224]);
[predictedLabel, scores] = classify(net, newImageResized);

fprintf('Prediction: %s
', predictedLabel);
fprintf('Class probabilities (Cancer vs Non-Cancer):
');
fprintf('    %.4f    %.4f
', scores(1), scores(2));

labelNames = net.Layers(end).Classes;
[~, idx] = max(scores);
predictedClass = labelNames(idx);
fprintf('This image is predicted to be %s with %.2f%% confidence.
', predictedClass, scores(idx) * 100);
disp('Classification completed successfully.');

position = [10 10];
labelStr = lower(string(predictedLabel));
if labelStr == "cancer" || labelStr == "cancers"
    labelText = 'Prediction: Cancer';
    textColor = [255 0 0];
else
    labelText = 'Prediction: Non-Cancer';
    textColor = [0 255 0];
end

annotatedImage = insertText(newImageResized, position, labelText, ...
    'FontSize', 20, 'BoxColor', 'black', 'TextColor', textColor, 'BoxOpacity', 0.6);

figure;
imshow(annotatedImage);
title('CT Scan Classification Result');
