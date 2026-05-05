origDir = fullfile('split_data_levels', 'test', 'clean');
enhRoot = 'enhanced_levels';
resultsDir = 'results';

if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

methods = { ...
    'blur_1','blur_2','blur_3', ...
    'low_light_1','low_light_2','low_light_3', ...
    'compressed_1','compressed_2','compressed_3'};

validExt = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};

allSummary = {};
row = 1;

for m = 1:length(methods)
    methodName = methods{m};
    methodPath = fullfile(enhRoot, methodName);

    if ~exist(methodPath, 'dir')
        continue;
    end

    files = dir(fullfile(methodPath, '*.*'));
    files = files(~[files.isdir]);

    psnrVals = [];
    ssimVals = [];
    y_true = [];
    y_pred = [];
    y_score = [];

    for i = 1:length(files)
        [~, name, ext] = fileparts(files(i).name);
        if ~ismember(lower(ext), validExt)
            continue;
        end

        origPath = fullfile(origDir, [name ext]);

        if ~exist(origPath, 'file')
            origCandidates = dir(fullfile(origDir, [name '.*']));
            origCandidates = origCandidates(~[origCandidates.isdir]);

            found = false;
            for j = 1:length(origCandidates)
                [~,~,e2] = fileparts(origCandidates(j).name);
                if ismember(lower(e2), validExt)
                    origPath = fullfile(origDir, origCandidates(j).name);
                    found = true;
                    break;
                end
            end

            if ~found
                continue;
            end
        end

        orig = imread(origPath);
        enh = imread(fullfile(methodPath, files(i).name));

        if size(orig,1) ~= size(enh,1) || size(orig,2) ~= size(enh,2)
            enh = imresize(enh, [size(orig,1), size(orig,2)]);
        end

        psnrVal = psnr(enh, orig);

        if size(orig,3) == 3
            origGray = rgb2gray(orig);
        else
            origGray = orig;
        end

        if size(enh,3) == 3
            enhGray = rgb2gray(enh);
        else
            enhGray = enh;
        end

        ssimVal = ssim(enhGray, origGray);

        psnrVals(end+1) = psnrVal; %#ok<AGROW>
        ssimVals(end+1) = ssimVal; %#ok<AGROW>

        % degraded = positive class
        y_true(end+1) = 1; %#ok<AGROW>
        y_score(end+1) = 1 - ssimVal; %#ok<AGROW>

        if ssimVal >= 0.80
            y_pred(end+1) = 0; %#ok<AGROW>
        else
            y_pred(end+1) = 1; %#ok<AGROW>
        end
    end

    cleanFiles = dir(fullfile(origDir, '*.*'));
    cleanFiles = cleanFiles(~[cleanFiles.isdir]);

    for i = 1:length(cleanFiles)
        [~,~,ext] = fileparts(cleanFiles(i).name);
        if ~ismember(lower(ext), validExt)
            continue;
        end
        y_true(end+1) = 0; %#ok<AGROW>
        y_pred(end+1) = 0; %#ok<AGROW>
        y_score(end+1) = 0; %#ok<AGROW>
    end

    cm = confusionmat(y_true, y_pred, 'Order', [0 1]);

    TN = cm(1,1);
    FP = cm(1,2);
    FN = cm(2,1);
    TP = cm(2,2);

    ACC = (TP + TN) / max(sum(cm(:)), 1);

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

    fig1 = figure('Visible', 'off');
    confusionchart(cm, {'clean','degraded'});
    title(['Confusion Matrix - ' methodName]);
    saveas(fig1, fullfile(resultsDir, ['robust_confusion_' methodName '.png']));
    close(fig1);

    fig2 = figure('Visible', 'off');
    plot(X, Y, 'LineWidth', 2);
    hold on;
    plot([0 1], [0 1], '--');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC Curve - ' methodName]);
    grid on;
    saveas(fig2, fullfile(resultsDir, ['robust_roc_' methodName '.png']));
    close(fig2);

    allSummary{row,1} = methodName;
    allSummary{row,2} = mean(psnrVals);
    allSummary{row,3} = mean(ssimVals);
    allSummary{row,4} = ACC;
    allSummary{row,5} = PRE;
    allSummary{row,6} = REC;
    allSummary{row,7} = F1;
    allSummary{row,8} = AUC;
    allSummary{row,9} = MCC;
    row = row + 1;
end

T = cell2table(allSummary, ...
    'VariableNames', {'Method','PSNR','SSIM','ACC','PRE','REC','F1','AUC','MCC'});

writetable(T, fullfile(resultsDir, 'robustness_metrics.csv'));
disp(T);
disp('Robustness metrics saved successfully.');