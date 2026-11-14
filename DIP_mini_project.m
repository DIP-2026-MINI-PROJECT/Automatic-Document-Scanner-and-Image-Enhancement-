
clear; clc; close all;


% IMAGE INPUT SETTINGS

forceDeblur = false;
BLUR_THRESHOLD = 0;
MAX_SIDE = 1600;
MIN_DOC_FRACTION = 0.10;

outDir = fullfile(pwd, 'output_matlab');

% Select image every time the program runs
[f, p] = uigetfile( ...
    {'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', ...
    'Supported Image Files'}, ...
    'Select Document Image');

if isequal(f, 0)
    error('No image selected.');
end

imagePath = fullfile(p, f);

fprintf('Selected image: %s\n', imagePath);

if ~license('test', 'Image_Toolbox')
    error('Image Processing Toolbox is required.');
end
if ~isfile(imagePath)
    [f, p] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif', 'Image files'}, ...
                       'Select the document photo');
    if isequal(f, 0), error('No image selected.'); end
    imagePath = fullfile(p, f);
end
if ~exist(outDir, 'dir'), mkdir(outDir); end
saveImg = @(name, x) imwrite(x, fullfile(outDir, name));

% Read and validate input image

try
    img = imread(imagePath);
catch ME
    error('Unable to read selected image: %s', ME.message);
end

% Convert image to uint8
img = im2uint8(img);

% Handle grayscale images
if size(img, 3) == 1
    img = repmat(img, [1 1 3]);
end

% Handle images with alpha channel
if size(img, 3) > 3
    img = img(:, :, 1:3);
end

% Resize large images
if max(size(img, 1), size(img, 2)) > MAX_SIDE
    img = imresize(img, ...
        MAX_SIDE / max(size(img, 1), size(img, 2)));
end
if max(size(img, 1), size(img, 2)) > MAX_SIDE
    img = imresize(img, MAX_SIDE / max(size(img, 1), size(img, 2)));
end

%  STAGE 1  

fprintf('=== STAGE 1: Units 1-2 (fundamentals + spatial enhancement) ===\n');

%  Image fundamentals: size, channels, class, bit depth
[h0, w0, c0] = size(img);
info = whos('img');
fprintf('Height x Width : %d x %d\n', h0, w0);
fprintf('Channels       : %d\n', c0);
fprintf('Data class     : %s\n', class(img));
fprintf('Bit depth      : %d-bit per channel\n', 8 * info.bytes / numel(img));
fprintf('Total pixels   : %d\n', h0 * w0);
fprintf('Min / Max value: %d / %d\n', min(img(:)), max(img(:)));

% Grayscale conversion (colour space transform)
grayImg = rgb2gray(img);

% Intensity (point) transformations
negImg   = imcomplement(grayImg);                         
g        = double(grayImg);
logImg   = uint8((255 / log(1 + max(g(:)))) * log(1 + g)); 
gammaImg = imadjust(grayImg, [], [], 0.6);                
stretchImg = imadjust(grayImg, stretchlim(grayImg, 0), [0 1]);  

% Histogram processing: global histogram equalization
eqImg = histeq(grayImg);

% Spatial smoothing filters (neighbourhood operations)
gaussImg  = imgaussfilt(grayImg, 1.1, 'FilterSize', 5);
medianImg = medfilt2(grayImg, [5 5]);

saveImg('s1_01_original.jpg', img);
saveImg('s1_02_gray.jpg', grayImg);
saveImg('s1_03_negative.jpg', negImg);
saveImg('s1_04_log.jpg', logImg);
saveImg('s1_05_gamma.jpg', gammaImg);
saveImg('s1_06_contrast_stretch.jpg', stretchImg);
saveImg('s1_07_hist_equalized.jpg', eqImg);
saveImg('s1_08_gaussian.jpg', gaussImg);
saveImg('s1_09_median.jpg', medianImg);

% Histogram before/after equalization
fig1 = figure('Name', 'Stage 1 - histogram', 'Position', [100 100 1000 400]);
subplot(1, 2, 1); imhist(grayImg); title('Histogram before equalization');
subplot(1, 2, 2); imhist(eqImg);   title('Histogram after equalization');
saveas(fig1, fullfile(outDir, 's1_histogram_comparison.png'));

% Side-by-side comparison figure (subplot 3 x 3)
fig2 = figure('Name', 'Stage 1 - comparison', 'Position', [100 50 1000 1000]);
panels1 = {img, 'Original'; grayImg, 'Grayscale'; negImg, 'Negative'; ...
           logImg, 'Log transform'; gammaImg, 'Gamma (0.6)'; ...
           stretchImg, 'Contrast stretch'; eqImg, 'Histogram equalized'; ...
           gaussImg, 'Gaussian 5x5'; medianImg, 'Median 5x5'};
for k = 1:9
    subplot(3, 3, k); imshow(panels1{k, 1}); title(panels1{k, 2});
end
saveas(fig2, fullfile(outDir, 's1_comparison.png'));


%  STAGE 2 
fprintf('\n=== STAGE 2: Units 3-4 (restoration, edges, morphology, segmentation) ===\n');

% 1. Restoration: blur detection + Wiener deconvolution
[restored, sBefore, sAfter, ~] = restoreImage(img, forceDeblur, BLUR_THRESHOLD);
saveImg('s2_02_restored.jpg', restored);
grayR = rgb2gray(restored);

% 2. CLAHE (contrast-limited adaptive histogram equalization)
claheGray = adapthisteq(grayR, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
saveImg('s2_03_clahe.jpg', claheGray);

fig3 = figure('Name', 'Stage 2 - CLAHE histogram', 'Position', [100 100 1000 400]);
subplot(1, 2, 1); imhist(grayR);      title('Histogram before CLAHE');
subplot(1, 2, 2); imhist(claheGray);  title('Histogram after CLAHE');
saveas(fig3, fullfile(outDir, 's2_histogram_clahe.png'));

% 3. Edge detection: Sobel and Canny
smoothImg = imgaussfilt(claheGray, 1.1, 'FilterSize', 5);
[Gmag, ~] = imgradient(smoothImg, 'sobel');
sobelImg  = mat2gray(Gmag);
cannyImg  = edge(smoothImg, 'canny');
saveImg('s2_04_sobel.jpg', sobelImg);
saveImg('s2_05_canny.jpg', cannyImg);

%  4+5. Morphology + segmentation: find the document
[quad, mask, method] = findDocument(grayR, cannyImg, MIN_DOC_FRACTION);
fprintf('Document found using: %s\n', method);
saveImg('s2_06_morph_mask.jpg', mask);

[hh, ww] = size(grayR);
polyMask = poly2mask(quad(:, 1), quad(:, 2), hh, ww);
edgeMask = imdilate(bwperim(polyMask), strel('disk', 3));
outline  = restored;
for ch = 1:3
    tmp = outline(:, :, ch);
    tmp(edgeMask) = (ch == 2) * 255;          
    outline(:, :, ch) = tmp;
end
docOnly = restored .* uint8(polyMask);       
saveImg('s2_07_document_outline.jpg', outline);
saveImg('s2_08_document_only.jpg', docOnly);

% 6. Geometric transformation: perspective warp
warped = fourPointWarp(restored, quad);
saveImg('s2_09_warped.jpg', warped);

% 7. Final enhancement
lab = rgb2lab(warped);                                        
lab(:, :, 1) = adapthisteq(lab(:, :, 1) / 100, ...
                           'ClipLimit', 0.02, 'NumTiles', [8 8]) * 100;
colorScan = lab2rgb(lab, 'OutputType', 'uint8');
saveImg('s2_10_scan_color.jpg', colorScan);

wg   = im2double(medfilt2(rgb2gray(warped), [3 3]));          
blk  = oddn(max(11, min(size(wg)) / 25));
sig  = 0.3 * ((blk - 1) * 0.5 - 1) + 0.8;
localMean = imgaussfilt(wg, sig, 'FilterSize', blk);
bwScan = wg > (localMean - 10 / 255);      
saveImg('s2_11_scan_bw.jpg', bwScan);

%  Summary figure
fig4 = figure('Name', 'Stage 2 - pipeline', 'Position', [50 50 1400 1000]);
panels2 = {img, '1. Input'; ...
           restored, sprintf('2. Restored (sharpness %.0f -> %.0f)', sBefore, sAfter); ...
           claheGray, '3. CLAHE'; sobelImg, '4. Sobel edges'; ...
           cannyImg, '5. Canny edges'; mask, '6. Morphology mask'; ...
           outline, '7. Document outline'; docOnly, '8. Document only'; ...
           warped, '9. Perspective warp'; colorScan, '10. Colour scan'; ...
           bwScan, '11. B&W scan'};
for k = 1:size(panels2, 1)
    subplot(3, 4, k); imshow(panels2{k, 1}); title(panels2{k, 2}, 'FontSize', 9);
end
saveas(fig4, fullfile(outDir, 's2_pipeline.png'));

fprintf('\nDone. Outputs saved in: %s\n', outDir);


function s = sharpness(im)

    if size(im, 3) == 3, g = rgb2gray(im); else, g = im; end
    g = imresize(g, 800 / size(g, 2));
    L = imfilter(double(g), fspecial('laplacian', 0), 'replicate');
    s = var(L(:));
end

function v = pct(x, p)

    x = sort(x(:));
    n = numel(x);
    k = min(max(round(p / 100 * n), 1), n);
    v = x(k);
end

function [extra, total] = estimateBlurSigma(im)

    g = imgaussfilt(double(rgb2gray(im)), 0.8);
    [gx, gy] = gradient(g);
    mag = hypot(gx, gy);
    A = pct(g, 95) - pct(g, 5);
    G = pct(mag, 99.9);
    total = A / (sqrt(2 * pi) * max(G, 1e-3));
    extra = sqrt(max(total^2 - 1.1^2, 0));
    extra = min(max(extra, 0.8), 5);
end

function out = wienerDeblur(im, sigma, nsr)

    pad = 40;
    f = padarray(im2double(im), [pad pad], 'symmetric');   
    hsize = 2 * ceil(3 * sigma) + 1;
    psf = fspecial('gaussian', hsize, sigma);
    for k = 1:size(f, 3)
        f(:, :, k) = deconvwnr(f(:, :, k), psf, nsr);
    end
    f = f(pad + 1:end - pad, pad + 1:end - pad, :);
    out = im2uint8(min(max(f, 0), 1));
end

function [out, before, after, sigma] = restoreImage(im, force, thr)

    before = sharpness(im);
    fprintf('Sharpness score (Laplacian variance): %.1f (threshold %d)\n', before, thr);
    sigma = NaN;
    if before >= thr && ~force
        fprintf('Image is sharp -> restoration skipped.\n');
        out = im; after = before;
        return;
    end
    [sigma, total] = estimateBlurSigma(im);
    fprintf(['Image is blurry -> estimated blur sigma = %.2f px ' ...
             '(edge-based total %.2f) -> Wiener deconvolution\n'], sigma, total);
    out = wienerDeblur(im, sigma, 0.001);
    after = sharpness(out);
    fprintf('Sharpness %.1f -> %.1f\n', before, after);
end

function n = oddn(x)
    n = round(x);
    if mod(n, 2) == 0, n = n + 1; end
end

function quad = cornersFromMask(mask)

    [ys, xs] = find(mask);
    s = xs + ys;  d = xs - ys;
    [~, iTL] = min(s);  [~, iBR] = max(s);
    [~, iTR] = max(d);  [~, iBL] = min(d);
    quad = [xs(iTL) ys(iTL); xs(iTR) ys(iTR); xs(iBR) ys(iBR); xs(iBL) ys(iBL)];
end

function [quad, mask, method] = findDocument(grayImg, edgesBW, minFrac)

    [h, w] = size(grayImg);
    imgArea = h * w;

    blurred = imgaussfilt(grayImg, 2);
    mask = imbinarize(blurred, graythresh(blurred));
    se = strel('square', oddn(max(9, 0.025 * min(h, w))));
    mask = imclose(mask, se);          
    mask = imopen(mask, se);           
    mask = imfill(mask, 'holes');     
    mask = bwareafilt(mask, 1);        
    frac = nnz(mask) / imgArea;
    if frac >= minFrac && frac <= 0.92
        quad = cornersFromMask(mask);
        method = 'Otsu threshold + morphology';
        return;
    end

    se1 = strel('square', 5);
    d = imdilate(imdilate(edgesBW, se1), se1);
    closed = imclose(d, strel('square', oddn(max(9, 0.02 * min(h, w)))));
    closed = bwareafilt(imfill(closed, 'holes'), 1);
    if nnz(closed) / imgArea >= minFrac
        quad = cornersFromMask(closed);
        mask = closed;
        method = 'Canny edges + morphology (backup)';
        return;
    end

    fprintf('No document found -> using the full frame.\n');
    quad = [1 1; w 1; w h; 1 h];
    method = 'none (full frame fallback)';
end

function warped = fourPointWarp(im, quad)

    tl = quad(1, :); tr = quad(2, :); br = quad(3, :); bl = quad(4, :);
    W = max(50, round(max(norm(br - bl), norm(tr - tl))));
    H = max(50, round(max(norm(tr - br), norm(tl - bl))));
    fixedPts = [1 1; W 1; W H; 1 H];
    tform = fitgeotrans(quad, fixedPts, 'projective');
    warped = imwarp(im, tform, 'OutputView', imref2d([H W]));
end