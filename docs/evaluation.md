# Final model evaluation (`BrainTumorFinal.pt`, threshold 0.5)

| Set | n (tumour / healthy) | Acc | Bal. acc | Precision | Recall | Specificity | F1 | ROC-AUC | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|---|---|
| leak_free_test | 599 (517 / 82) | 0.937 | 0.938 | 0.990 | 0.936 | 0.939 | 0.962 | 0.968 | 77 / 5 / 33 / 484 |
| testing_benchmark | 1600 (1200 / 400) | 0.916 | 0.902 | 0.957 | 0.929 | 0.875 | 0.943 | 0.964 | 350 / 50 / 85 / 1115 |
| external_253 | 253 (155 / 98) | 0.822 | 0.847 | 0.966 | 0.735 | 0.959 | 0.835 | 0.942 | 94 / 4 / 41 / 114 |
| leak_free_test_original_source | 242 (198 / 44) | 0.913 | 0.912 | 0.978 | 0.914 | 0.909 | 0.945 | 0.938 | 40 / 4 / 17 / 181 |
| leak_free_test_new_source | 357 (319 / 38) | 0.952 | 0.962 | 0.997 | 0.950 | 0.974 | 0.973 | 0.992 | 37 / 1 / 16 / 303 |

Latency (CPU, 8 threads, 60 images): model 1.95 ms (p95 2.12), preprocess+model 2.58 ms, full web pipeline 15.93 ms (p95 27.99).
Memory: weights 1.52 MB, 393,618 parameters, process RSS after load 274.2 MB.

external_253 is held out from the training pool (near-duplicates removed) but comes from the ORIGINAL source, not an independent hospital.

## Independent healthy test: IXI (new source, never used for training)

IXI T1 scans of 581 healthy volunteers from three London hospitals (https://brain-development.org/ixi-dataset/, CC BY-SA 3.0), 3 axial slices each, 30, 45 and 60 mm below the top of the head (`experiments/ixi/make_slices.py`, `evaluate_ixi.py`, results in `docs/ixi_evaluation.json`). All scans are healthy, so only false tumour calls can be measured.

| Metric | Result |
|---|---|
| Slice specificity (healthy correctly cleared) | **0.956** (1,667 / 1,743) |
| False tumour calls | 76 slices, in 75 of 581 subjects |
| Reliability: false tumour calls sent to human review | **35 of 76 (46%)** |
| Reliability: correct answers sent to human review | 57 of 1,667 (3.4%) |

A wrong answer was about 13 times more likely to be flagged for review than a correct one. This set has no tumour scans, so it says nothing about sensitivity on a new source.
