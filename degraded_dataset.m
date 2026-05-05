inputDir = 'dataset/images';
outputRoot = 'dataset_quality';

folders = {'clean','blur','low_light','compressed'};
for i = 1:length(folders)
    outDir = fullfile(outputRoot, folders{i});
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
end

files = dir(fullfile(inputDir, '*.*'));
files = files(~[files.isdir]);

for i = 1:length(files)
    fileName = files(i).name;
    img = imread(fullfile(inputDir, fileName));

    imwrite(img, fullfile(outputRoot, 'clean', fileName));

    blur = imgaussfilt(img, 2);
    imwrite(blur, fullfile(outputRoot, 'blur', fileName));

    low = uint8(double(img) * 0.35);
    imwrite(low, fullfile(outputRoot, 'low_light', fileName));

    tempPath = 'temp.jpg';
    imwrite(img, tempPath, 'Quality', 15);
    comp = imread(tempPath);
    imwrite(comp, fullfile(outputRoot, 'compressed', fileName));
    delete(tempPath);
end