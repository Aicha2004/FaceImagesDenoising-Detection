inputDir = 'dataset/images';
outputRoot = 'dataset_quality_levels';

classes = { ...
    'clean', ...
    'blur_1', 'blur_2', 'blur_3', ...
    'low_light_1', 'low_light_2', 'low_light_3', ...
    'compressed_1', 'compressed_2', 'compressed_3'};

if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
end

for i = 1:length(classes)
    outDir = fullfile(outputRoot, classes{i});
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
end

files = dir(fullfile(inputDir, '*.*'));
files = files(~[files.isdir]);

validExt = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};

for i = 1:length(files)
    [~, name, ext] = fileparts(files(i).name);

    if ~ismember(lower(ext), validExt)
        continue;
    end

    img = imread(fullfile(inputDir, files(i).name));

    % clean
    imwrite(img, fullfile(outputRoot, 'clean', [name ext]));

    % blur levels
    b1 = imgaussfilt(img, 1);
    b2 = imgaussfilt(img, 2);
    b3 = imgaussfilt(img, 4);

    imwrite(b1, fullfile(outputRoot, 'blur_1', [name ext]));
    imwrite(b2, fullfile(outputRoot, 'blur_2', [name ext]));
    imwrite(b3, fullfile(outputRoot, 'blur_3', [name ext]));

    % low-light levels
    l1 = im2uint8(im2double(img) * 0.70);
    l2 = im2uint8(im2double(img) * 0.45);
    l3 = im2uint8(im2double(img) * 0.25);

    imwrite(l1, fullfile(outputRoot, 'low_light_1', [name ext]));
    imwrite(l2, fullfile(outputRoot, 'low_light_2', [name ext]));
    imwrite(l3, fullfile(outputRoot, 'low_light_3', [name ext]));

    % compression levels
    tmp1 = fullfile(outputRoot, 'tmp_q30.jpg');
    tmp2 = fullfile(outputRoot, 'tmp_q15.jpg');
    tmp3 = fullfile(outputRoot, 'tmp_q5.jpg');

    imwrite(img, tmp1, 'Quality', 30);
    imwrite(img, tmp2, 'Quality', 15);
    imwrite(img, tmp3, 'Quality', 5);

    c1 = imread(tmp1);
    c2 = imread(tmp2);
    c3 = imread(tmp3);

    imwrite(c1, fullfile(outputRoot, 'compressed_1', [name '.jpg']));
    imwrite(c2, fullfile(outputRoot, 'compressed_2', [name '.jpg']));
    imwrite(c3, fullfile(outputRoot, 'compressed_3', [name '.jpg']));
end

tmpFiles = {'tmp_q30.jpg','tmp_q15.jpg','tmp_q5.jpg'};
for i = 1:length(tmpFiles)
    f = fullfile(outputRoot, tmpFiles{i});
    if exist(f, 'file')
        delete(f);
    end
end

disp('dataset_quality_levels created successfully.');