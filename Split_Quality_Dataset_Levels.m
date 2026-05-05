srcRoot = 'dataset_quality_levels';
dstRoot = 'split_data_levels';

classes = { ...
    'clean', ...
    'blur_1', 'blur_2', 'blur_3', ...
    'low_light_1', 'low_light_2', 'low_light_3', ...
    'compressed_1', 'compressed_2', 'compressed_3'};

trainRatio = 0.70;
valRatio   = 0.15;
testRatio  = 0.15;

for c = 1:length(classes)
    className = classes{c};
    srcDir = fullfile(srcRoot, className);

    if ~exist(srcDir, 'dir')
        continue;
    end

    files = dir(fullfile(srcDir, '*.*'));
    files = files(~[files.isdir]);

    validNames = {};
    validExt = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};

    for i = 1:length(files)
        [~,~,ext] = fileparts(files(i).name);
        if ismember(lower(ext), validExt)
            validNames{end+1} = files(i).name; %#ok<AGROW>
        end
    end

    n = numel(validNames);
    if n == 0
        continue;
    end

    rng(42);
    idx = randperm(n);

    nTrain = round(trainRatio * n);
    nVal = round(valRatio * n);

    trainIdx = idx(1:nTrain);
    valIdx = idx(nTrain+1:nTrain+nVal);
    testIdx = idx(nTrain+nVal+1:end);

    splits = {'train','val','test'};
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

disp('split_data_levels created successfully.');