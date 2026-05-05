srcDir = 'dataset/images';
outDir = 'split_data_binary';

trainRatio = 0.70;
valRatio = 0.15;

validExt = {'.jpg', '.jpeg', '.png', '.bmp'};

splits = {'train','val','test'};

for s = 1:length(splits)
    folderPath = fullfile(outDir, splits{s}, 'clean');
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
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

copyList(trainFiles, srcDir, fullfile(outDir,'train','clean'));
copyList(valFiles, srcDir, fullfile(outDir,'val','clean'));
copyList(testFiles, srcDir, fullfile(outDir,'test','clean'));

disp(['Train: ', num2str(length(trainFiles))]);
disp(['Val: ', num2str(length(valFiles))]);
disp(['Test: ', num2str(length(testFiles))]);
disp('Clean dataset split completed.');

function copyList(fileList, srcDir, dstDir)
    for k = 1:length(fileList)
        srcPath = fullfile(srcDir, fileList{k});
        dstPath = fullfile(dstDir, fileList{k});
        if ~exist(dstPath, 'file')
            copyfile(srcPath, dstPath);
        end
    end
end