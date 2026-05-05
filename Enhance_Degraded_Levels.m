inputRoot = fullfile('split_data_levels', 'test');
outputRoot = 'enhanced_levels';

degradedClasses = { ...
    'blur_1','blur_2','blur_3', ...
    'low_light_1','low_light_2','low_light_3', ...
    'compressed_1','compressed_2','compressed_3'};

for i = 1:length(degradedClasses)
    outDir = fullfile(outputRoot, degradedClasses{i});
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
end

validExt = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'};

for c = 1:length(degradedClasses)
    className = degradedClasses{c};
    inDir = fullfile(inputRoot, className);

    if ~exist(inDir, 'dir')
        continue;
    end

    files = dir(fullfile(inDir, '*.*'));
    files = files(~[files.isdir]);

    for i = 1:length(files)
        [~, name, ext] = fileparts(files(i).name);
        if ~ismember(lower(ext), validExt)
            continue;
        end

        img = imread(fullfile(inDir, files(i).name));

        if startsWith(className, 'blur')
            out = imsharpen(img, 'Radius', 2, 'Amount', 1.5);

        elseif startsWith(className, 'low_light')
            if size(img,3) == 3
                lab = rgb2lab(img);
                L = lab(:,:,1) / 100;
                L = adapthisteq(L, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
                lab(:,:,1) = L * 100;
                out = lab2rgb(lab);
                out = im2uint8(out);
            else
                out = adapthisteq(img);
            end

        elseif startsWith(className, 'compressed')
            out = imbilatfilt(img);

        else
            out = img;
        end

        imwrite(out, fullfile(outputRoot, className, [name ext]));
    end
end

disp('enhanced_levels created successfully.');