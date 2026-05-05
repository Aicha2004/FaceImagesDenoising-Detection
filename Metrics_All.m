origDir = fullfile('split_data', 'test', 'clean');
resultsDir = 'results';
enhRoot = 'enhanced';

if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

methods = {'blur_fixed', 'low_light_fixed', 'compressed_fixed'};
sourceClass = {'blur', 'low_light', 'compressed'};

allSummary = {};
summaryRow = 1;
allPerImage = {};

validExt = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};

for m = 1:length(methods)
    methodName = methods{m};
    methodPath = fullfile(enhRoot, methodName);

    if ~exist(methodPath, 'dir')
        warning('Missing folder: %s', methodPath);
        continue;
    end

    files = dir(fullfile(methodPath, '*.*'));
    files = files(~[files.isdir]);

    y_true = [];
    y_pred = [];
    y_score = [];
    psnrVals = [];
    ssimVals = [];
    methodPerImage = {};
    perRow = 1;

    % positive class: degraded/non-clean = 1
    for i = 1:length(files)
        [~, name, ext] = fileparts(files(i).name);
        if ~ismember(lower(ext), validExt)
            continue;
        end

        origPath = fullfile(origDir, [name ext]);

        % fallback if extension changed during compression
        if ~exist(origPath, 'file')
            origCandidates = dir(fullfile(origDir, [name '.*']));
            origCandidates = origCandidates(~[origCandidates.isdir]);
            found = false;
            for t = 1:length(origCandidates)
                [~,~,e2] = fileparts(origCandidates(t).name);
                if ismember(lower(e2), validExt)
                    origPath = fullfile(origDir, origCandidates(t).name);
                    found = true;
                    break;
                end
            end
            if ~found
                continue;
            end
        end

        denPath = fullfile(methodPath, files(i).name);

        orig = imread(origPath);
        den = imread(denPath);

        if size(orig,1) ~= size(den,1) || size(orig,2) ~= size(den,2)
            den = imresize(den, [size(orig,1), size(orig,2)]);
        end

        psnrVal = psnr(den, orig);

        if size(orig,3) == 3
            origGray = rgb2gray(orig);
        else
            origGray = orig;
        end

        if size(den,3) == 3
            denGray = rgb2gray(den);
        else
            denGray = den;
        end

        ssimVal = ssim(denGray, origGray);

        psnrVals(end+1) = psnrVal; %#ok<AGROW>
        ssimVals(end+1) = ssimVal; %#ok<AGROW>

        % degraded input class is true positive class
        y_true(end+1) = 1; %#ok<AGROW>

        % score for positive class: more degradation = higher score
        score = 1 - ssimVal;
        y_score(end+1) = score; %#ok<AGROW>

        % prediction rule
        if ssimVal >= 0.80
            predLabel = 0;
        else
            predLabel = 1;
        end
        y_pred(end+1) = predLabel; %#ok<AGROW>

        methodPerImage{perRow,1} = methodName;
        methodPerImage{perRow,2} = [name ext];
        methodPerImage{perRow,3} = psnrVal;
        methodPerImage{perRow,4} = ssimVal;
        methodPerImage{perRow,5} = score;
        methodPerImage{perRow,6} = 1;
        methodPerImage{perRow,7} = predLabel;
        perRow = perRow + 1;
    end

    % add clean test images as negative class
    cleanFiles = dir(fullfile(origDir, '*.*'));
    cleanFiles = cleanFiles(~[cleanFiles.isdir]);

    for i = 1:length(cleanFiles)
        [~, ~, ext] = fileparts(cleanFiles(i).name);
        if ~ismember(lower(ext), validExt)
            continue;
        end

        y_true(end+1) = 0; %#ok<AGROW>
        y_pred(end+1) = 0; %#ok<AGROW>
        y_score(end+1) = 0; %#ok<AGROW>
    end

    if isempty(y_true)
        warning('No data for method: %s', methodName);
        continue;
    end

    cm = confusionmat(y_true, y_pred, 'Order', [0 1]);

    TN = cm(1,1);
    FP = cm(1,2);
    FN = cm(2,1);
    TP = cm(2,2);

    total = sum(cm(:));
    ACC = (TP + TN) / max(total, 1);

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

    [X, Y, ~, AUC] = perfcurve(y_true, y_score, 1);

    % save per-image csv
    if ~isempty(methodPerImage)
        T_per = cell2table(methodPerImage, ...
            'VariableNames', {'Method','Image','PSNR','SSIM','Score','TrueLabel','PredLabel'});
        writetable(T_per, fullfile(resultsDir, ['per_image_' methodName '.csv']));
        allPerImage = [allPerImage; methodPerImage]; %#ok<AGROW>
    end

    % save confusion matrix image
    fig1 = figure('Visible', 'off');
    confusionchart(cm, {'clean','degraded'});
    title(['Confusion Matrix - ' methodName]);
    saveas(fig1, fullfile(resultsDir, ['confusion_matrix_' methodName '.png']));
    close(fig1);

    % save roc curve
    fig2 = figure('Visible', 'off');
    plot(X, Y, 'LineWidth', 2);
    hold on;
    plot([0 1], [0 1], '--');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC Curve - ' methodName]);
    grid on;
    saveas(fig2, fullfile(resultsDir, ['roc_curve_' methodName '.png']));
    close(fig2);

    allSummary{summaryRow,1} = methodName;
    allSummary{summaryRow,2} = mean(psnrVals);
    allSummary{summaryRow,3} = mean(ssimVals);
    allSummary{summaryRow,4} = ACC;
    allSummary{summaryRow,5} = PRE;
    allSummary{summaryRow,6} = REC;
    allSummary{summaryRow,7} = F1;
    allSummary{summaryRow,8} = AUC;
    allSummary{summaryRow,9} = MCC;
    summaryRow = summaryRow + 1;
end

if ~isempty(allSummary)
    T_summary = cell2table(allSummary, ...
        'VariableNames', {'Method','PSNR','SSIM','ACC','PRE','REC','F1','AUC','MCC'});
    writetable(T_summary, fullfile(resultsDir, 'metrics_summary.csv'));
    disp(T_summary);
end

if ~isempty(allPerImage)
    T_all = cell2table(allPerImage, ...
        'VariableNames', {'Method','Image','PSNR','SSIM','Score','TrueLabel','PredLabel'});
    writetable(T_all, fullfile(resultsDir, 'metrics_per_image_all.csv'));
end

disp('All metrics saved successfully.');