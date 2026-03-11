# tasks/fault_detection/  ← FUTURE TASK

This folder does not exist yet. When you are ready to build fault detection,
the only files you need to create are:

```
tasks/fault_detection/
├── feature_set.py    ← define FEATURES, TARGET = "fault_any"
├── config.yaml       ← model_type: xgboost_classifier, eval_metric: f1
├── train.py          ← from training.runner import run_training (same pattern as soc_forecast)
├── evaluate.py       ← F1, AUC, confusion matrix using evaluation/metrics.py
└── README.md
```

Then run:
```bash
python run.py train --task fault_detection
python run.py eval  --task fault_detection --dt 2026-04-01
```

Nothing in ingestion/, features/, training/, evaluation/, or monitoring/ changes.
The fault model gets its own versioned run folder in models/fault_detection/.

Wire it into run.py by adding two lines:
```python
# in cmd_train:
elif task == "fault_detection":
    from tasks.fault_detection.train import run_training
    run_training()
```
