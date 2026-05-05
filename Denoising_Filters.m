inputRoot = 'noisyMatlab';
outputRoot = 'denoisedMatlab';

if ~exist(outputRoot, 'dir')
    mkdir(outputRoot);
end

filterNames = {'gaussian_filter', 'median_filter', 'min_filter', 'max_filter', 'adaptive_filter'};

for i = 1:length(filterNames)
    outDir = fullfile(outputRoot, filterNames{i});
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
end

noiseFolders = dir(inputRoot);
noiseFolders = noiseFolders([noiseFolders.isdir]);
noiseFolders = noiseFolders(~ismember({noiseFolders.name}, {'.', '..'}));

for n = 1:length(noiseFolders)
    noiseType = noiseFolders(n).name;
    inDir = fullfile(inputRoot, noiseType);

    files = dir(fullfile(inDir, '*.*'));
    files = files(~[files.isdir]);

    for i = 1:length(files)
        fileName = files(i).name;
        imgPath = fullfile(inDir, fileName);
        img = imread(imgPath);

        if size(img,3) == 3
            gray = rgb2gray(img);
        else
            gray = img;
        end

        % 1. Gaussian filter
        g = imgaussfilt(img, 2);
        imwrite(g, fullfile(outputRoot, 'gaussian_filter', [noiseType '_' fileName]));

        % 2. Median filter
        if size(img,3) == 3
            m = img;
            for c = 1:3
                m(:,:,c) = medfilt2(img(:,:,c), [3 3]);
            end
        else
            m = medfilt2(img, [3 3]);
        end
        imwrite(m, fullfile(outputRoot, 'median_filter', [noiseType '_' fileName]));

        % 3. Min filter
        if size(img,3) == 3
            minf = img;
            for c = 1:3
                minf(:,:,c) = ordfilt2(img(:,:,c), 1, true(3));
            end
        else
            minf = ordfilt2(img, 1, true(3));
        end
        imwrite(minf, fullfile(outputRoot, 'min_filter', [noiseType '_' fileName]));

        % 4. Max filter
        if size(img,3) == 3
            maxf = img;
            for c = 1:3
                maxf(:,:,c) = ordfilt2(img(:,:,c), 9, true(3));
            end
        else
            maxf = ordfilt2(img, 9, true(3));
        end
        imwrite(maxf, fullfile(outputRoot, 'max_filter', [noiseType '_' fileName]));

        % 5. Adaptive Wiener filter
        if size(img,3) == 3
            af = img;
            for c = 1:3
                af(:,:,c) = uint8(wiener2(img(:,:,c), [5 5]));
            end
        else
            af = uint8(wiener2(img, [5 5]));
        end
        imwrite(af, fullfile(outputRoot, 'adaptive_filter', [noiseType '_' fileName]));
    end
end

disp('Denoised images saved successfully.');