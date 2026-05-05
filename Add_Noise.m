inputDir = 'dataset/images';
outputRoot = 'noisyMatlab';

if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
end

noiseTypes = {'gaussian', 'salt_pepper', 'speckle', 'poisson', 'motion'};

for i = 1:length(noiseTypes)
    outDir = fullfile(outputRoot, noiseTypes{i});
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
end

files = dir(fullfile(inputDir, '*.*'));
files = files(~[files.isdir]);

for i = 1:length(files)
    fileName = files(i).name;
    imgPath = fullfile(inputDir, fileName);
    img = imread(imgPath);

    % 1. Gaussian noise
    noisy_gaussian = imnoise(img, 'gaussian', 0, 0.01);
    imwrite(noisy_gaussian, fullfile(outputRoot, 'gaussian', fileName));

    % 2. Salt & pepper noise
    noisy_sp = imnoise(img, 'salt & pepper', 0.02);
    imwrite(noisy_sp, fullfile(outputRoot, 'salt_pepper', fileName));

    % 3. Speckle noise
    noisy_speckle = imnoise(img, 'speckle', 0.04);
    imwrite(noisy_speckle, fullfile(outputRoot, 'speckle', fileName));

    % 4. Poisson noise
    noisy_poisson = imnoise(img, 'poisson');
    imwrite(noisy_poisson, fullfile(outputRoot, 'poisson', fileName));

    % 5. Motion blur noise
    h = fspecial('motion', 10, 45);
    noisy_motion = imfilter(img, h, 'replicate');
    imwrite(noisy_motion, fullfile(outputRoot, 'motion', fileName));
end

disp('Noise images saved successfully.');