inputRoot = fullfile('split_data', 'test');
outputRoot = 'enhanced';

folders = {'blur_fixed', 'low_light_fixed', 'compressed_fixed'};
for i = 1:length(folders)
    folderPath = fullfile(outputRoot, folders{i});
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end
end

validExt = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};

% ---------------- blur -> sharpen ----------------
blurDir = fullfile(inputRoot, 'blur');
if exist(blurDir, 'dir')
    files = dir(fullfile(blurDir, '*.*'));
    files = files(~[files.isdir]);

    for i = 1:length(files)
        [~, name, ext] = fileparts(files(i).name);
        if ~ismember(lower(ext), validExt)
            continue;
        end

        img = imread(fullfile(blurDir, files(i).name));
        out = imsharpen(img, 'Radius', 2, 'Amount', 1.5);
        imwrite(out, fullfile(outputRoot, 'blur_fixed', [name ext]));
    end
end

% ---------------- low light -> CLAHE ----------------
lowDir = fullfile(inputRoot, 'low_light');
if exist(lowDir, 'dir')
    files = dir(fullfile(lowDir, '*.*'));
    files = files(~[files.isdir]);

    for i = 1:length(files)
        [~, name, ext] = fileparts(files(i).name);
        if ~ismember(lower(ext), validExt)
            continue;
        end

        img = imread(fullfile(lowDir, files(i).name));

        if size(img, 3) == 3
            lab = rgb2lab(img);
            L = lab(:,:,1) / 100;
            L = adapthisteq(L, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
            lab(:,:,1) = L * 100;
            out = lab2rgb(lab);
            out = im2uint8(out);
        else
            out = adapthisteq(img);
        end

        imwrite(out, fullfile(outputRoot, 'low_light_fixed', [name ext]));
    end
end

% ---------------- compressed -> bilateral filter ----------------
compDir = fullfile(inputRoot, 'compressed');
if exist(compDir, 'dir')
    files = dir(fullfile(compDir, '*.*'));
    files = files(~[files.isdir]);

    for i = 1:length(files)
        [~, name, ext] = fileparts(files(i).name);
        if ~ismember(lower(ext), validExt)
            continue;
        end

        img = imread(fullfile(compDir, files(i).name));
        out = imbilatfilt(img);
        imwrite(out, fullfile(outputRoot, 'compressed_fixed', [name ext]));
    end
end

disp('enhanced images created successfully.');