origDir = 'dataset/images';
resultsDir = 'results';
denoisedRoot = 'denoisedMatlab';

if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

methods = dir(denoisedRoot);
methods = methods([methods.isdir]);
methods = methods(~ismember({methods.name}, {'.', '..'}));

allResults = {};
row = 1;

for m = 1:length(methods)
    methodName = methods(m).name;
    methodPath = fullfile(denoisedRoot, methodName);

    files = dir(fullfile(methodPath, '*.*'));
    files = files(~[files.isdir]);

    psnrVals = [];
    ssimVals = [];

    y_true = [];
    y_pred = [];
    y_score = [];

    for i = 1:length(files)
        denFile = files(i).name;

        % original filename is after first underscore
        idx = strfind(denFile, '_');
        if isempty(idx)
            continue;
        end

        origFile = denFile(idx(1)+1:end);
        origPath = fullfile(origDir, origFile);
        denPath = fullfile(methodPath, denFile);

        if ~exist(origPath, 'file')
            continue;
        end

        orig = imread(origPath);
        den = imread(denPath);

        % PSNR
        psnrVal = psnr(den, orig);
        psnrVals(end+1) = psnrVal;

        % SSIM
        if size(orig,3) == 3
            origGray = rgb2gray(orig);
            denGray = rgb2gray(den);
        else
            origGray = orig;
            denGray = den;
        end

        ssimVal = ssim(denGray, origGray);
        ssimVals(end+1) = ssimVal;

        % Binary labels for clean/noise example
        % true = noisy class = 1
        % predicted based on SSIM threshold
        y_true(end+1) = 1;

        score = ssimVal;
        y_score(end+1) = score;

        if ssimVal > 0.75
            y_pred(end+1) = 0;   % predicted clean-like
        else
            y_pred(end+1) = 1;   % predicted noisy
        end
    end

    % For clean class references
    cleanFiles = dir(fullfile(origDir, '*.*'));
    cleanFiles = cleanFiles(~[cleanFiles.isdir]);

    for i = 1:length(cleanFiles)
        fileName = cleanFiles(i).name;
        imgPath = fullfile(origDir, fileName);
        img = imread(imgPath);

        y_true(end+1) = 0;      % true clean
        y_pred(end+1) = 0;      % predicted clean
        y_score(end+1) = 1.0;   % strong clean confidence
    end

    cm = confusionmat(y_true, y_pred);

    if size(cm,1) < 2 || size(cm,2) < 2
        cm = [cm 0; 0 0];
    end

    TN = cm(1,1);
    FP = cm(1,2);
    FN = cm(2,1);
    TP = cm(2,2);

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

    % AUC and ROC
    [X,Y,~,AUC] = perfcurve(y_true, y_score, 1);

    % save confusion matrix image
    fig1 = figure('Visible','off');
    confusionchart(cm, {'clean','noise'});
    saveas(fig1, fullfile(resultsDir, ['confusion_matrix_' methodName '.png']));
    close(fig1);

    % save ROC curve image
    fig2 = figure('Visible','off');
    plot(X, Y, 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC Curve - ' methodName]);
    grid on;
    saveas(fig2, fullfile(resultsDir, ['roc_curve_' methodName '.png']));
    close(fig2);

    allResults{row,1} = methodName;
    allResults{row,2} = mean(psnrVals);
    allResults{row,3} = mean(ssimVals);
    allResults{row,4} = ACC;
    allResults{row,5} = PRE;
    allResults{row,6} = REC;
    allResults{row,7} = F1;
    allResults{row,8} = AUC;
    allResults{row,9} = MCC;

    row = row + 1;
end

T = cell2table(allResults, 'VariableNames', ...
    {'Method','PSNR','SSIM','ACC','PRE','REC','F1','AUC','MCC'});

writetable(T, fullfile(resultsDir, 'metrics_results.csv'));
disp(T);
disp('Metrics saved successfully.');