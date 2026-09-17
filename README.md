# GB Glove Extractor

Separates the insulating glove from the background. Image in, mask out (255 = glove, same size as the input).
The mask is deliberately a little larger than the glove, so defects on the glove edge are never cut off.
Method: `median_gamma_cut_multicc_safe`.

## Install

```bash
pip install -r requirements.txt
```

## Use

```bash
python glove_extractor.py frame.bmp mask.png      # one image
python glove_extractor.py frames/ masks/          # a whole folder
```

```python
import cv2
from glove_extractor import extract

mask = extract(cv2.imread("frame.bmp"))                # parameters from params.json
mask = extract(cv2.imread("frame.bmp"), dilate_px=20)  # any keyword overrides params.json
```

## Parameters (`params.json`)

The image is first downscaled by `scale`; every size below is in pixels at that scale.

| key | meaning |
|---|---|
| `scale` | working resolution (0.5 of a 2448x2048 frame) |
| `stretch_lo`, `stretch_hi` | percentiles for contrast stretching |
| `blur_sigma` | blur before thresholding |
| `open_k`, `close_k` | clean-up of the thresholded mask |
| `mask_sever_k`, `cut_sever_k` | break thin links to background objects |
| `extra_cc_frac`, `extra_cc_max` | close-ups: keep extra fingers touching the bottom edge when the largest piece is below this share of the frame (0 disables) |
| `median_k`, `gamma` | edge preprocessing |
| `canny_sigma`, `canny_pct`, `canny_ratio`, `edge_thick` | edge detection |
| `guard_lo`, `guard_hi` | an edge cut is kept only if it removes this share of the mask |
| `dilate_px` | safety margin added around the glove |

Tuned on 2448x2048 frames from the glove inspection rig (about 0.07 s per frame). If you change a value, re-check the result on labelled images.
