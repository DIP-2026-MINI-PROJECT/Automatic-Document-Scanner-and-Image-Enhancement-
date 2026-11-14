%% DIP MINI PROJECT: BATCH RUN (MATLAB)
% Runs the document scanner on every image in the "data" folder and saves,
% for each image, the detected outline and the final scan in "outputs".
%
% How to run: set MATLAB's Current Folder to the repository folder (the one
% that contains the "data" folder), open this file and press Run.
% Needs: MATLAB + Image Processing Toolbox.

clear; clc; close all;

dataDir = fullfile(pwd, 'data');
outDir  = fullfile(pwd, 'outputs');
if ~exist(dataDir, 'dir')
    error('Put the input images in a folder named "data" inside the current folder.');
end
if ~exist(outDir, 'dir'), mkdir(outDir); end

files = [dir(fullfile(dataDir, '*.jpg')); dir(fullfile(dataDir, '*.jpeg')); ...
         dir(fullfile(dataDir, '*.png'))];
fprintf('%d images found in %s\n\n', numel(files), dataDir);

for k = 1:numel(files)
    [~, name] = fileparts(files(k).name);
    img = imread(fullfile(dataDir, files(k).name));
    tic; r = scanDocument(img); t = toc;
    imwrite(r.outline, fullfile(outDir, [name '_outline.jpg']));
    imwrite(r.final,   fullfile(outDir, [name '_scan.jpg']));
    fprintf('%-30s %5.2f s   %s\n', files(k).name, t, r.method);
end
fprintf('\nDone. Outputs saved in: %s\n', outDir);

%% ======================================================================
%  LOCAL FUNCTIONS (must stay at the end of the file)
%  ======================================================================

function r = scanDocument(img)
% SCANDOCUMENT  Edge-based document scanner (DIP mini project).
%   r = scanDocument(img) takes an RGB photo of a document on a surface and
%   returns a struct with every intermediate result and the final scan.
%
%   Pipeline: grayscale -> Gaussian smoothing -> CLAHE -> Sobel / Canny ->
%   keep long edges -> 4 corners -> perspective correction -> final CLAHE.
%   Needs the Image Processing Toolbox.

    WORK_SIZE        = 1600;   % longer side is resized to this
    SIGMA            = 5;      % Gaussian sigma (31 x 31 filter)
    MIN_EDGE_LENGTH  = 0.30;   % edge must span >= 30% of the image diagonal
    MIN_DOC_FRACTION = 0.10;   % document covers at least 10% of the frame
    MAX_DOC_FRACTION = 0.95;   % ... and is not the whole frame
    MIN_SUPPORT      = 0.30;   % share of the outline lying on detected edges
    % Canny [low high] thresholds (fractions of the maximum gradient),
    % tried in this order until a valid document outline is found
    CANNY_LIST = [0.04 0.12; 0.06 0.18; 0.09 0.27; 0.02 0.06; 0.13 0.38; 0.01 0.03];

    img = im2uint8(img);
    if size(img, 3) == 1, img = repmat(img, [1 1 3]); end
    s = WORK_SIZE / max(size(img, 1), size(img, 2));
    img = imresize(img, s);
    [h, w, ~] = size(img);
    r.scale = s;
    r.input = img;

    % ---- 1. Grayscale (Unit 1)
    gray = rgb2gray(img);
    r.gray = gray;

    % ---- 2. Gaussian smoothing with a large sigma (Unit 2)
    smooth = imgaussfilt(gray, SIGMA, 'FilterSize', 2 * ceil(3 * SIGMA) + 1);
    r.smooth = smooth;

    % ---- 3. CLAHE (Unit 3)
    clahe = adapthisteq(smooth, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
    r.clahe = clahe;

    % ---- 4a. Sobel gradient magnitude, for comparison (Unit 3)
    [Gmag, ~] = imgradient(clahe, 'sobel');
    r.sobel = mat2gray(Gmag);

    % ---- 4b, 5, 6. Canny edges -> keep long edges -> 4 corners (Units 3, 4)
    quad = [];
    canny = false(h, w);
    longEdges = false(h, w);
    method = '';
    for k = 1:size(CANNY_LIST, 1)
        e = edge(clahe, 'canny', CANNY_LIST(k, :), 1);
        L = keepLongEdges(e, MIN_EDGE_LENGTH);
        q = cornersFromEdges(L);
        if k == 1, canny = e; longEdges = L; end      % something to show if all fail
        if isempty(q), continue; end
        frac = polyarea(q(:, 1), q(:, 2)) / (h * w);
        sup = edgeSupport(q, imdilate(e, strel('square', 7)));
        if frac >= MIN_DOC_FRACTION && frac <= MAX_DOC_FRACTION && sup >= MIN_SUPPORT
            quad = q; canny = e; longEdges = L;
            method = sprintf('4-corner outline from long edges (Canny thresholds [%.2f %.2f])', ...
                             CANNY_LIST(k, 1), CANNY_LIST(k, 2));
            break;
        end
    end
    if isempty(quad)
        quad = [1 1; w 1; w h; 1 h];
        method = 'fallback: full frame';
    end
    r.canny = canny;
    r.longEdges = longEdges;
    r.quad = quad;
    r.method = method;

    % ---- outline drawn on the image (green border, red corners)
    polyMask = poly2mask(quad(:, 1), quad(:, 2), h, w);
    edgeMask = imdilate(bwperim(polyMask), strel('square', 5));
    outline = img;
    for ch = 1:3
        tmp = outline(:, :, ch);
        tmp(edgeMask) = 255 * (ch == 2);
        outline(:, :, ch) = tmp;
    end
    for i = 1:4
        x = round(quad(i, 1)); y = round(quad(i, 2));
        xr = max(1, x - 7):min(w, x + 7);
        yr = max(1, y - 7):min(h, y + 7);
        outline(yr, xr, 1) = 255; outline(yr, xr, 2) = 0; outline(yr, xr, 3) = 0;
    end
    r.outline = outline;

    % ---- 7. Perspective (projective) transformation (Unit 4)
    warped = fourPointWarp(img, quad);
    r.warped = warped;

    % ---- 8. Final enhancement: CLAHE on the lightness channel
    lab = rgb2lab(warped);
    lab(:, :, 1) = 100 * adapthisteq(lab(:, :, 1) / 100, 'ClipLimit', 0.02, 'NumTiles', [8 8]);
    r.final = lab2rgb(lab, 'OutputType', 'uint8');
end

function keep = keepLongEdges(edges, minFrac)
% Edge linking by length: dilate, label 8-connected pieces, keep the long ones.
    [h, w] = size(edges);
    linked = imdilate(edges, strel('square', 5));
    cc = bwconncomp(linked, 8);
    keep = false(h, w);
    if cc.NumObjects == 0, return; end
    stats = regionprops(cc, 'BoundingBox');
    bb = vertcat(stats.BoundingBox);                  % [x y width height]
    len = hypot(bb(:, 3), bb(:, 4));
    idx = find(len >= minFrac * hypot(h, w));
    if isempty(idx), return; end
    keep = ismember(labelmatrix(cc), idx);
end

function q = cornersFromEdges(L)
% Convex hull of the long edges -> 4 corners (TL, TR, BR, BL).
    q = [];
    [ys, xs] = find(L);
    if numel(xs) < 50, return; end
    try
        k = convhull(xs, ys);
    catch
        return;
    end
    hx = xs(k); hy = ys(k);
    s = hx + hy; d = hx - hy;
    [~, iTL] = min(s); [~, iBR] = max(s);
    [~, iTR] = max(d); [~, iBL] = min(d);
    q = [hx(iTL) hy(iTL); hx(iTR) hy(iTR); hx(iBR) hy(iBR); hx(iBL) hy(iBL)];
    if size(unique(q, 'rows'), 1) < 4, q = []; end
end

function sup = edgeSupport(q, supportMap)
% Fraction of points along the 4 sides of the outline that lie on edges.
    [h, w] = size(supportMap);
    hits = 0; total = 0;
    for i = 1:4
        a = q(i, :); b = q(mod(i, 4) + 1, :);
        t = linspace(0, 1, 200)';
        x = round(a(1) + (b(1) - a(1)) * t);
        y = round(a(2) + (b(2) - a(2)) * t);
        ok = x >= 1 & x <= w & y >= 1 & y <= h;
        idx = sub2ind([h w], y(ok), x(ok));
        hits = hits + sum(supportMap(idx));
        total = total + nnz(ok);
    end
    sup = hits / max(total, 1);
end

function warped = fourPointWarp(im, quad)
% Perspective correction: 4 document corners -> upright rectangle.
    tl = quad(1, :); tr = quad(2, :); br = quad(3, :); bl = quad(4, :);
    W = max(50, round(max(norm(br - bl), norm(tr - tl))));
    H = max(50, round(max(norm(tr - br), norm(tl - bl))));
    fixedPts = [1 1; W 1; W H; 1 H];
    tform = fitgeotrans(quad, fixedPts, 'projective');
    warped = imwarp(im, tform, 'OutputView', imref2d([H W]));
end
