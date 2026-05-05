srcRoot = 'dataset_quality';
dstRoot = 'split_data';

classes = {'clean', 'blur', 'low_light', 'compressed'};

trainRatio = 0.70;
valRatio   = 0.15;
testRatio  = 0.15;

if abs(trainRatio + valRatio + testRatio - 1.0) > 1e-6
    error('Train/Val/Test ratios must sum to 1.');
end

for c = 1:length(classes)
    className = classes{c};
    srcDir = fullfile(srcRoot, className);

    if ~exist(srcDir, 'dir')
        warning('Missing folder: %s', srcDir);
        continue;
    end

    files = dir(fullfile(srcDir, '*.*'));
    files = files(~[files.isdir]);

    validNames = {};
    validExt = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};

    for i = 1:length(files)
        [~, ~, ext] = fileparts(files(i).name);
        if ismember(lower(ext), validExt)
            validNames{end+1} = files(i).name; %#ok<AGROW>
        end
    end

    n = numel(validNames);
    if n == 0
        warning('No valid images in %s', srcDir);
        continue;
    end

    rng(42);
    idx = randperm(n);

    nTrain = round(trainRatio * n);
    nVal   = round(valRatio * n);
    nTest  = n - nTrain - nVal;

    trainIdx = idx(1:nTrain);
    valIdx   = idx(nTrain+1:nTrain+nVal);
    testIdx  = idx(nTrain+nVal+1:end);

    splits = {'train', 'val', 'test'};
    splitIndices = {trainIdx, valIdx, testIdx};

    for s = 1:length(splits)
        outDir = fullfile(dstRoot, splits{s}, className);
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end

        currentIdx = splitIndices{s};
        for k = 1:length(currentIdx)
            fname = validNames{currentIdx(k)};
            copyfile(fullfile(srcDir, fname), fullfile(outDir, fname));
        end
    end
end

disp('split_data created successfully.');