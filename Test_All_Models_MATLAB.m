clc; clear; close all;

baseDir = 'C:\Users\dell\Desktop\project';

dataDir = fullfile(baseDir, 'split_data_binary');
modelDir = fullfile(baseDir, 'models_binary_matlab');
resultsDir = fullfile(baseDir, 'results_binary_matlab');

if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

imgSize = [224 224 3];

testDir = fullfile(dataDir, 'test');

testDS = imageDatastore(testDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

augTest = augmentedImageDatastore(imgSize, testDS);

modelFiles = dir(fullfile(modelDir, '*.mat'));

results = {};
row = 1;

for i = 1:length(modelFiles)

    modelPath = fullfile(modelDir, modelFiles(i).name);
    S = load(modelPath);

    if isfield(S, 'trainedNet')
        model = S.trainedNet;
    elseif isfield(S, 'net')
        model = S.net;
    else
        warning('No network variable found in %s. Skipping...', modelFiles(i).name);
        continue;
    end

    modelName = erase(modelFiles(i).name, {'best_', '.mat'});

    fprintf('\nTesting model: %s\n', modelName);

    [predLabels, scores] = classify(model, augTest);
    trueLabels = testDS.Labels;

    cm = confusionmat(trueLabels, predLabels);
    classNames = categories(trueLabels);

    cleanIdx = find(strcmp(classNames, 'clean'));
    noiseIdx = find(strcmp(classNames, 'noise'));

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

    results{row,1} = modelName;
    results{row,2} = ACC;
    results{row,3} = PRE;
    results{row,4} = REC;
    results{row,5} = F1;
    results{row,6} = AUC;
    results{row,7} = MCC;

    cmTable = array2table(cm, ...
        'VariableNames', strcat('Pred_', string(classNames)), ...
        'RowNames', strcat('True_', string(classNames)));

    writetable(cmTable, ...
        fullfile(resultsDir, ['confusion_matrix_' modelName '.csv']), ...
        'WriteRowNames', true);

    fig = figure('Visible','off');
    cmChart = confusionchart(trueLabels, predLabels);
    cmChart.Title = ['Confusion Matrix - ' modelName];
    cmChart.RowSummary = 'row-normalized';
    cmChart.ColumnSummary = 'column-normalized';

    exportgraphics(fig, ...
        fullfile(resultsDir, ['confusion_matrix_' modelName '.png']), ...
        'Resolution', 300);

    close(fig);

    row = row + 1;
end

if isempty(results)
    warning('No models were tested.');
else
    T = cell2table(results, ...
        'VariableNames', {'Model','ACC','PRE','REC','F1','AUC','MCC'});

    writetable(T, fullfile(resultsDir, 'all_model_metrics_matlab.csv'));

    disp(T);
end

disp('All MATLAB model metrics saved.');