# MRI VisionGuard: app flow

One page, no navigation. Everything happens top to bottom, and each step reveals
the next only when it has something to show. All copy uses British spelling; the
API's "Tumor" / "No tumor" values are displayed as "Tumour" / "No tumour".

## Journey

1. **Landing.** Sticky translucent header (logo mark, name, model chips). Short
   intro and the safety notice, then one card with the samples up front: four
   normal samples ("Try a sample") and, in a separate outlined group, two difficult
   scans ("Try a difficult scan": Low contrast, which runs the follow-up test and
   holds, and Very dark, which ends in "Poor input quality"). A compact one-line
   drop zone (click or drag) sits under them. At 1280x800 the samples are above
   the fold. Below: "Reading the heatmap", "Limitations", "About the model".
2. **Upload or sample.** The file is checked in the browser (JPEG/PNG, 10 MB)
   before it is sent. Problems show inline under the drop zone, never as a pop-up.
3. **Analysing.** A small status row; the previous result is hidden so old and
   new answers never mix.
4. **Results**, two columns on desktop:
   - **Right (sticky, stays in view while scrolling):** the verdict card
     (prediction, status badge, one-line meaning, both class scores as bars
     with a 50% threshold tick and the winning row emphasised, "not calibrated
     probabilities" note, "Analyse another scan") and the Reliability panel
     (initial vs follow-up score and change when a follow-up ran, reliability
     warnings, decision trace, the "research-demo state" note).
   - When the status is not "consistent", the verdict carries it: the badge
     moves above the prediction, the prediction and bars turn muted, and a
     caveat (low confidence, or "Flagged for human review: <status>") sits
     under the meaning line, before the bars.
   - The decision trace folds like Image quality: open from 861 px up, closed
     on narrower screens. On phones the quality table drops its "Expected"
     column.
   - **Left:** the Grad-CAM viewer (toggle, strength slider, drag-to-compare
     divider), "What the model sees" (original upload, cropped square, model
     input as three large thumbnails), and Image quality.
   - The viewer uses the API's `content_box` to trim the square padding: the
     viewer has the scan's own aspect ratio and clips a frame that holds the
     scan and heatmap at identical size and offset, so they stay aligned.
   - Image quality shows a one-line summary ("All 9 checks passed" / "2 of 9
     checks need attention"); the 9-row table is folded away and opens by
     itself only when a check has status "warning".
   - Cards materialise in reading order (opacity + slight scale/blur, 300 ms;
     fade only with reduced motion).
5. **Again.** "Analyse another scan" scrolls back to the top and opens the file
   picker.

## Wireframes

Desktop (>= 860 px)

```
+--------------------------------------------------------------+
| (o) MRI VisionGuard        [393,618 params][128x128][PyTorch]|  sticky, blurred
+--------------------------------------------------------------+
| Explainable MRI analysis ...           (safety notice)       |
| Try a sample [s1][s2][s3][s4]   | Try a difficult scan [d1][d2]|
| [ ^ Or drop your own MRI image here, or click to choose ]    |
+-------------------------------------+------------------------+
| Grad-CAM viewer  [x] heatmap        | VERDICT  Tumour [badge]|  sticky
| [ trimmed scan + compare divider ]  |  bars, not calibrated  |  column
| strength ----o----   legend         | [Analyse another scan] |
+-------------------------------------+------------------------+
| What the model sees                 | Reliability            |
| [ Original ] [ Crop ] [ Model ]     |  initial -> follow-up  |
+-------------------------------------+  warnings, trace       |
| Image quality  (All 9 checks passed)|                        |
+-------------------------------------+------------------------+
| Reading the heatmap | Limitations | About the model           |
+--------------------------------------------------------------+
```

Phone (390 px): single column, verdict first.

```
+------------------------+
| (o) MRI VisionGuard    |  sticky
+------------------------+
| intro + notice         |
| [s1][s2][s3][s4]       |
| difficult: [d1][d2]    |
| [ drop zone ]          |
| VERDICT + badge + bars |
| Reliability + trace    |
| Grad-CAM viewer        |
| Original | Crop | Model|
| Image quality summary  |
| info cards             |
+------------------------+
```
