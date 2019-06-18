%using the files on matlab file exchange to retrieve
%images as matrix and labels as vector

%NOTE: to avoid running all steps you can simply load the 'vars.mat' file
%in too the workspace and begin in section after training is completed

% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
 %%
for i = 1:size(images,2)
im = (images(:,i));
im = reshape(im, 28, 28);
t = strcat(num2str(i), '.png');
folder = strcat(num2str(labels(i)),'/');
folder = strcat('Numbers/', folder);
t = strcat(folder, t);
imwrite(im,t)
end
%%
%imshow(im);
%bag = bagOfFeatures(images)
imds = imageDatastore('Numbers','IncludeSubfolders',true,'LabelSource','foldernames');
save('imds', 'imds')
%%
[trainingSet, validationSet] = splitEachLabel(imds, 0.6, 'randomize');
bag = bagOfFeatures(trainingSet);

save('bag', 'bag')
%%
img = readimage(imds, 1);
featureVector = encode(bag, img);
figure;imshow(img)

% Plot the histogram of visual word occurrences
figure
bar(featureVector)
title('occurrences')
xlabel('Visual feature index')
ylabel('Frequency of occurrence')
%%
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag);
save('categoryClassifier', 'categoryClassifier')

%%
confMatrix = evaluate(categoryClassifier, validationSet)
save('confMatrix', 'confMatrix')
%%
T = table(confMatrix(:,1),confMatrix(:,2),confMatrix(:,3),confMatrix(:,4),confMatrix(:,5),confMatrix(:,6),confMatrix(:,7),...
    confMatrix(:,8),confMatrix(:,9),confMatrix(:,10),'variableNames',{'predict0','predict1','predict2','predict3', 'predict4','predict5','predict6','predict7','predict8','predict9'});
T.Properties.RowNames = {'Actual0' 'Actual1' 'Actual2' 'Actual3' 'Actual4' 'Actual5' 'Actual6' 'Actual7' 'Actual8' 'Actual9' }
table
%%
%find a validation image that the model gets wrong 
found = false;
i = 1;
hardnums = [];
while i < length(validationSet.Files)
 
img =readimage(validationSet,i);
[labelIdx, scores] = predict(categoryClassifier, img);
guess = labelIdx-1;
a = validationSet.Labels(i);
if (categorical(guess) ~= a )
 hardnums = horzcat(hardnums, i)
end

i = i+1;
end
montage(validationSet.Files(hardnums))
%%
a = hardnums(1);
newTrain = trainingSet;
newTrain.Files{end+1} = validationSet.Files{a};
newTrain.Labels(length(newTrain.Files),1) = '0';
%%
newBag = bagOfFeatures(trainingSet);
%%
newCategoryClassifier = trainImageCategoryClassifier(newTrain, newBag);
save('newCategoryClassifier', 'newCategoryClassifier')
%%
img =readimage(validationSet,hardnums(1));
imshow(img)
[labelIdx, scores] = predict(newCategoryClassifier, img);
guess = labelIdx-1
%%
newConfMatrix = evaluate(newCategoryClassifier, validationSet)
%%
T = table(newConfMatrix(:,1),newConfMatrix(:,2),newConfMatrix(:,3),newConfMatrix(:,4),newConfMatrix(:,5),newConfMatrix(:,6),newConfMatrix(:,7),...
    newConfMatrix(:,8),newConfMatrix(:,9),newConfMatrix(:,10),'variableNames',{'predict0','predict1','predict2','predict3', 'predict4','predict5','predict6','predict7','predict8','predict9'});
T.Properties.RowNames = {'Actual0' 'Actual1' 'Actual2' 'Actual3' 'Actual4' 'Actual5' 'Actual6' 'Actual7' 'Actual8' 'Actual9' }


