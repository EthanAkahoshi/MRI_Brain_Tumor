# MRI VisionGuard: app flow

One page, no navigation. Everything happens top to bottom, and each step reveals
the next only when it has something to show.

## Journey

1. **Landing.** Sticky translucent header (name, model chips). Short intro, the
   safety notice, then one upload card: drop zone (click or drag) and four sample
   scans. Below: "Reading the heatmap", "Limitations", "About the model".
2. **Upload or sample.** The file is checked in the browser (JPEG/PNG, 10 MB)
   before it is sent. Problems show inline under the drop zone, never as a pop-up.
3. **Analysing.** A small status row replaces nothing and blocks nothing; the
   previous result is hidden so old and new answers never mix.
4. **Results** appear in three layers, in reading order, each one materialising
   a beat after the one before (opacity + slight scale/blur, 300 ms; fade only
   with reduced motion):
   1. **Verdict** (answer first): prediction, both class scores as bars, the
      reliability status badge, and a one-line plain-English meaning. A
      "scores are not calibrated probabilities" note sits right under the bars.
   2. **Evidence**: Grad-CAM viewer (toggle, strength slider, drag-to-compare
      divider) next to the image strip: original upload, cropped square, and
      the 128x128 model input, side by side.
   3. **Details**: Quality dashboard (9 checks, value, expected range, ok or
      warning) and Reliability dashboard (status, initial vs follow-up score and
      change when a follow-up ran, reliability warnings, decision trace as a
      numbered step list). The quality list starts open with a one-line summary
      ("All 9 checks passed") and can be collapsed.
5. **Again.** "Analyse another scan" scrolls back to the top and opens the file
   picker.

## Wireframes

Desktop (>= 860 px)

```
+--------------------------------------------------------------+
| MRI VisionGuard            [393,618 params][128x128][PyTorch]|  sticky, blurred
+--------------------------------------------------------------+
| Explainable MRI analysis ...           (safety notice)       |
| [ drop zone .............................. ] [s1][s2][s3][s4]|
+--------------------------------------------------------------+
| VERDICT  Tumor            [Consistent experimental result]   |
|          Tumour   ██████████████ 99.9%                       |
|          No tumour ▏            0.0%   scores not calibrated |
+-------------------------------------+------------------------+
| Grad-CAM viewer  [x] heatmap        | Original | Crop | Model|
| [ image with compare divider ]      | What the model sees    |
| strength ----o----   legend         | [Analyse another scan] |
+-------------------------------------+------------------------+
| Image quality (all 9 passed) v    | Reliability            |
|  ok Sharpness   349   >= 80         |  badge, initial->follow|
|  .. 9 rows                          |  1 Initial 2 Decision..|
+-------------------------------------+------------------------+
| Reading the heatmap | Limitations | About the model           |
+--------------------------------------------------------------+
```

Phone (390 px): single column, same order.

```
+------------------------+
| MRI VisionGuard        |  sticky
+------------------------+
| intro + notice         |
| [ drop zone ]          |
| [s1][s2][s3][s4]       |
| VERDICT + badge + bars |
| Grad-CAM viewer        |
| Original | Crop | Model|
| Quality checks    |
| Reliability + trace    |
| [Analyse another scan] |
| info cards             |
+------------------------+
```
