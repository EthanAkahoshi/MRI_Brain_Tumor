# MRI VisionGuard: demo video script (2–3 min)

Explainable MRI analysis and reliability monitoring. OpenCV AI Competition 2026, by Ethan.
Spoken text is about 440 words, roughly 2:45 at a calm pace.

## Prep before recording

- Start the app (`.venv\Scripts\python -m uvicorn webapp.app:app`) and open http://127.0.0.1:8000 at desktop width (860 px or wider).
- **Make the darkened image:** open one of the sample scans in any image editor, turn brightness down hard (and contrast down a little), and save it as a PNG (for example `dark_sample.png`) on the desktop. Upload it once before recording and check that the quality panel shows a Brightness or Contrast warning and the decision trace says "Contrast normalisation selected". If it doesn't, darken it more. Write down the status badge it gets (consistent or preprocessing-sensitive) and say the one you actually see.
- Have `docs/evaluation.md` open in a second tab or window for the numbers beat. You can also have `docs/gradcam_baseline.png` and `docs/gradcam_final.png` ready.
- Do one run of every beat first. Then hard-refresh so the page starts clean.

## Script

| Time | On screen | Ethan says |
|---|---|---|
| 0:00–0:20 | Landing page: header, intro, safety notice. | "A brain MRI classifier can score above ninety percent and still be cheating. Mine was. It had learned that a large, dark 512-pixel scan means tumour, because that's how one of the datasets was put together. MRI VisionGuard is what I built after I found that: a classifier that shows its evidence, checks its own input and tells you when not to trust it." |
| 0:20–0:45 | Click a tumour sample. The verdict appears: label, two score bars, status badge, and the note saying scores aren't calibrated. | "This is one of the built-in samples. The answer comes first: the predicted class, a score for each class, and a reliability status. These scores are not calibrated probabilities. A 99 means the model is confident, not that there's a 99 percent chance. The full pipeline takes about 16 milliseconds on a CPU, and the model is 1.5 megabytes." |
| 0:45–1:05 | Evidence panel: drag the compare divider over the image, move the strength slider. Point at the strip: original, crop, 128×128 input. | "Grad-CAM shows which regions pushed the score. I can drag across to compare it with the scan. It's a coarse heatmap, not a segmentation: it does not outline the tumour. On the right is what the model actually sees: the brain cropped out, padded square, brightness normalised, at 128 by 128." |
| 1:05–1:25 | Details: quality panel, "All 9 checks passed". Point at one row: value and expected range. | "Next to that are nine OpenCV checks: sharpness, brightness, contrast, saturated pixels, the brain crop, background, resolution, colour and stray marks. Each threshold is the 1st or 99th percentile of my training images. So a warning means 'unusual compared to what the model learned from'. It isn't a clinical quality grade." |
| 1:25–1:55 | Upload `dark_sample.png`. Brightness/contrast warning in the quality panel. The reliability panel shows the initial vs follow-up score and the numbered decision trace. | "Here's the same kind of scan, darkened. The quality check flags it. Then a second pass runs. It's a fixed rule table, not an agent. It picks at most one follow-up: here, contrast normalisation with OpenCV's CLAHE. It re-runs the model once and compares the score, the label and the heatmap. The first answer is the one reported. The follow-up only says whether that answer held up, and every step is written down in this trace." |
| 1:55–2:25 | Switch to `docs/evaluation.md`; optionally show the two Grad-CAM sheets side by side. | "Here's how I got here. I found the shortcut by predicting labels from image size and brightness alone. My 'external' test set turned out to be mostly copies of my training images. So I rebuilt the split with near-duplicate grouping and trained with source-balanced sampling. The honest numbers: 93.8 percent balanced accuracy on the leak-free test, 90.2 on the benchmark. On the held-out external set, sensitivity is only 73.5 percent. It still misses tumours there." |
| 2:25–2:45 | Back to the app; scroll to the Limitations card. | "The limits: the split is image-level, not patient-level, and the healthy set is small. The external set comes from the same source, not a different hospital. This is a research demo, not a medical device. What I'd like you to take from it is the process: find the shortcut, remove the leaks, and report the numbers as they are." |

## Numbers used (all from `docs/evaluation.md`)

- Leak-free test: balanced accuracy 0.938, AUC 0.968 (n = 599, 82 healthy).
- Testing benchmark: balanced accuracy 0.902 (n = 1,600).
- external_253: balanced accuracy 0.847, sensitivity 0.735, specificity 0.959. This set is held out but comes from the original source.
- Latency: 15.93 ms full web pipeline, 1.95 ms model only (CPU). Weights 1.52 MB, 393,618 parameters.

## Things not to claim

- That it diagnoses, detects or localises tumours, or that it's clinically validated. Grad-CAM is not a segmentation.
- That the scores are probabilities or confidence percentages in a calibrated sense.
- That the second pass is an "AI agent", is autonomous, or learns. It's a fixed rule table, and it never changes the answer.
- That the external set is independent or from another hospital, or that results are patient-level.
- That the 96 percent target was met. It wasn't on any set.
- Any number that isn't in `docs/evaluation.md`. Don't quote the old, leaky 94.8 / 96 percent figures as results.
- That the quality thresholds are clinical standards. They are percentiles of the training data.
