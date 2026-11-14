# Automatic Document Scanner and Image Enhancement

Digital Image Processing mini project, PES University (UE24EC342AC1)
Author: Sanidhya C M (PES1UG24EC290)

## Overview
Takes a photo of a document on a table and produces a straight, enhanced scan of only the document. The document is found automatically from its edges, the perspective is corrected, and the result is contrast-enhanced.

## Pipeline
1. Grayscale conversion
2. Gaussian smoothing (large sigma)
3. CLAHE
4. Edge detection (Sobel and Canny)
5. Edge linking: keep only the long edges (the page border)
6. Find the four document corners
7. Perspective correction
8. Final enhancement (CLAHE on the lightness channel)

## DIP topics and MATLAB functions
| Unit | Topics | Functions |
|---|---|---|
| 1 | Grayscale, negative, log, gamma, contrast stretching | `rgb2gray`, `imcomplement`, `imadjust` |
| 2 | Histogram equalization, Gaussian and median filtering | `histeq`, `imgaussfilt`, `medfilt2` |
| 3 | CLAHE, Sobel, Canny | `adapthisteq`, `imgradient`, `edge` |
| 4 | Dilation, edge linking, outline, perspective transform | `imdilate`, `bwconncomp`, `convhull`, `fitgeotrans`, `imwarp` |

## Requirements
- MATLAB
- Image Processing Toolbox

## How to run
1. Clone the repository:
   `git clone https://github.com/DIP-2026-MINI-PROJECT/Automatic-Document-Scanner-and-Image-Enhancement-.git`
2. Open `DIP_mini_project.m` in MATLAB.
3. Click **Run**. If the default photo is not found, a file picker opens; choose a document photo.
4. Results are saved in the `output_matlab` folder.

## Output
- Stage 1 (Units 1-2): grayscale, negative, log, gamma, contrast stretch, histogram equalization, Gaussian and median results, histogram and comparison figures
- Stage 2 (Units 3-4): smoothed image, CLAHE, Sobel, Canny, long edges, detected outline, perspective-corrected image, final scan

## Limitations
- The page needs some contrast against the surface
- One roughly rectangular document per photo
- Blur is not removed

## Repository contents
- `DIP_mini_project.m`: the full MATLAB code
- `Team23_report.docx`: project report
