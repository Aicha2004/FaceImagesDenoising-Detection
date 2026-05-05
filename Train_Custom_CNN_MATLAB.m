baseDir = 'C:\Users\dell\Desktop\project';

dataDir = fullfile(baseDir, 'split_data_binary');
resultsDir = fullfile(baseDir, 'results_binary_matlab');
modelDir = fullfile(baseDir, 'models_binary_matlab');

if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

if ~exist(modelDir, 'dir')
    mkdir(modelDir);
end

imgSize = [224 224 3];
batchSize = 16;
maxEpochs = 10;
learnRate = 1e-4;

trainDir = fullfile(dataDir, 'train');
valDir   = fullfile(dataDir, 'val');
testDir  = fullfile(dataDir, 'test');

trainDS = imageDatastore(trainDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

valDS = imageDatastore(valDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

testDS = imageDatastore(testDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

classes = categories(trainDS.Labels);
numClasses = numel(classes);

disp('Classes:');
disp(classes);

if numClasses < 2
    error('You need at least two classes: clean and noise.');
end

augTrain = augmentedImageDatastore(imgSize, trainDS, ...
    'DataAugmentation', imageDataAugmenter( ...
        'RandXReflection', true, ...
        'RandRotation', [-10 10]));

augVal = augmentedImageDatastore(imgSize, valDS);
augTest = augmentedImageDatastore(imgSize, testDS);

layers = [
    imageInputLayer(imgSize, 'Name', 'input')

    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')

    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool4')

    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool5')

    dropoutLayer(0.4, 'Name', 'dropout')

    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'InitialLearnRate', learnRate, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', batchSize, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

net = trainNetwork(augTrain, layers, options);

save(fullfile(modelDir, 'custom_cnn.mat'), 'net', 'classes');

[predLabels, scores] = classify(net, augTest);
trueLabels = testDS.Labels;

cm = confusionmat(trueLabels, predLabels);
classNames = categories(trueLabels);

cleanIdx = find(strcmp(classNames, 'clean'));
noiseIdx = find(strcmp(classNames, 'noise'));

if isempty(cleanIdx) || isempty(noiseIdx)
    error('Classes clean and noise must exist in the test folder.');
end

TN = cm(cleanIdx, cleanIdx);
FP = cm(cleanIdx, noiseIdx);
FN = cm(noiseIdx, cleanIdx);
TP = cm(noiseIdx, noiseIdx);

ACC = (TP + TN) / sum(cm(:));

if (TP + FP) == 0
    PRE = 0;
else
    PRE = TP / (TP + FP);
end

if (TP + FN) == 0
    REC = 0;
else
    REC = TP / (TP + FN);
end

if (PRE + REC) == 0
    F1 = 0;
else
    F1 = 2 * PRE * REC / (PRE + REC);
end

denom = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
if denom == 0
    MCC = 0;
else
    MCC = ((TP*TN) - (FP*FN)) / denom;
end

trueBinary = double(trueLabels == 'noise');
noiseScore = scores(:, noiseIdx);

try
    [~,~,~,AUC] = perfcurve(trueBinary, noiseScore, 1);
catch
    AUC = 0;
end

T = table(ACC, PRE, REC, F1, AUC, MCC);
writetable(T, fullfile(resultsDir, 'custom_cnn_metrics.csv'));

cmTable = array2table(cm, ...
    'VariableNames', strcat('Pred_', string(classNames)), ...
    'RowNames', strcat('True_', string(classNames)));

writetable(cmTable, fullfile(resultsDir, 'custom_cnn_confusion_matrix.csv'), ...
    'WriteRowNames', true);

fig = figure('Visible','off');
cmChart = confusionchart(trueLabels, predLabels);
cmChart.Title = 'Custom CNN Confusion Matrix';
cmChart.RowSummary = 'row-normalized';
cmChart.ColumnSummary = 'column-normalized';

exportgraphics(fig, fullfile(resultsDir, 'custom_cnn_confusion_matrix.png'), ...
    'Resolution', 300);

close(fig);

metricsNames = {'ACC','PRE','REC','F1','AUC','MCC'};
metricsValues = [ACC PRE REC F1 AUC MCC];

fig2 = figure('Visible','off');
bar(metricsValues);
set(gca, 'XTickLabel', metricsNames);
ylim([0 1]);
ylabel('Value');
title('Custom CNN Metrics');
grid on;

exportgraphics(fig2, fullfile(resultsDir, 'custom_cnn_metrics_bar.png'), ...
    'Resolution', 300);

close(fig2);

disp(T);
disp('Saved model:');
disp(fullfile(modelDir, 'custom_cnn.mat'));

disp('Saved metrics:');
disp(fullfile(resultsDir, 'custom_cnn_metrics.csv'));

disp('Saved confusion matrix image:');
disp(fullfile(resultsDir, 'custom_cnn_confusion_matrix.png'));

disp('Custom CNN training and evaluation completed.');