baseDir = 'C:\Users\dell\Desktop\project';

srcDir = fullfile(baseDir, 'dataset', 'images');
outDir = fullfile(baseDir, 'split_data_binary');

trainRatio = 0.70;
valRatio = 0.15;

validExt = {'.jpg', '.jpeg', '.png', '.bmp'};

splits = {'train','val','test'};
classes = {'clean','noise'};

for s = 1:length(splits)
    for c = 1:length(classes)
        folderPath = fullfile(outDir, splits{s}, classes{c});
        if ~exist(folderPath, 'dir')
            mkdir(folderPath);
        end
    end
end

files = dir(fullfile(srcDir, '*.*'));
files = files(~[files.isdir]);

validFiles = {};
for i = 1:length(files)
    [~,~,ext] = fileparts(files(i).name);
    if ismember(lower(ext), validExt)
        validFiles{end+1} = files(i).name; %#ok<SAGROW>
    end
end

rng(42);
idx = randperm(length(validFiles));
validFiles = validFiles(idx);

n = length(validFiles);
nTrain = round(trainRatio * n);
nVal = round(valRatio * n);

trainFiles = validFiles(1:nTrain);
valFiles = validFiles(nTrain+1:nTrain+nVal);
testFiles = validFiles(nTrain+nVal+1:end);

copyAndNoise(trainFiles, srcDir, fullfile(outDir,'train','clean'), fullfile(outDir,'train','noise'));
copyAndNoise(valFiles, srcDir, fullfile(outDir,'val','clean'), fullfile(outDir,'val','noise'));
copyAndNoise(testFiles, srcDir, fullfile(outDir,'test','clean'), fullfile(outDir,'test','noise'));

disp(['Train: ', num2str(length(trainFiles))]);
disp(['Val: ', num2str(length(valFiles))]);
disp(['Test: ', num2str(length(testFiles))]);
disp('Clean/noise dataset created successfully.');

function copyAndNoise(fileList, srcDir, cleanDir, noiseDir)

    for k = 1:length(fileList)
        fileName = fileList{k};

        srcPath = fullfile(srcDir, fileName);
        cleanPath = fullfile(cleanDir, fileName);
        noisePath = fullfile(noiseDir, fileName);

        img = imread(srcPath);

        if ~exist(cleanPath, 'file')
            copyfile(srcPath, cleanPath);
        end

        if ~exist(noisePath, 'file')
            noisyImg = addRandomNoise(img);
            imwrite(noisyImg, noisePath);
        end
    end
end

function noisyImg = addRandomNoise(img)

    noiseTypes = {'gaussian','salt_pepper','speckle','poisson','motion'};
    noiseType = noiseTypes{randi(length(noiseTypes))};

    imgDouble = im2double(img);

    switch noiseType

        case 'gaussian'
            noisyImg = imnoise(imgDouble, 'gaussian', 0, 0.01);

        case 'salt_pepper'
            noisyImg = imnoise(imgDouble, 'salt & pepper', 0.02);

        case 'speckle'
            noisyImg = imnoise(imgDouble, 'speckle', 0.04);

        case 'poisson'
            noisyImg = imnoise(imgDouble, 'poisson');

        case 'motion'
            h = fspecial('motion', 9, 0);
            noisyImg = imfilter(imgDouble, h, 'replicate');

        otherwise
            noisyImg = imgDouble;
    end

    noisyImg = im2uint8(noisyImg);
end