# CMMTS-hub (Based on Self-Developed Cross-Medium Multimodal Tactile Sensor)

Cross-medium grasp force forecasting with **CM-TFNet** (Cross-Medium Temporal Fusion Network).

The network is a next-frame predictor: given a sliding multimodal window `(B, T, 15)`, it outputs `[Fz_L, Fz_R]`. The reported protocol is **autoregressive rollout**. During lift-to-exit, predicted Fz is written back into the window; proximity, IMU, and medium state `s(t)` stay observed. Training matches this with scheduled sampling and multi-step unroll.

This repository contains only the **dataset** and **model code** (including ablations and baselines). Trained weights and experimental results are not included.

## Layout

```
CMMTS-hub/
├── cm_tfnet/                 # model, training, evaluation, ablations, baselines
│   ├── model.py              # CM-TFNet
│   ├── ablations.py          # architectural ablations
│   ├── baselines.py          # LSTM / GRU / TCN baselines
│   ├── train_rollout.py      # rollout training (scheduled sampling + unroll)
│   ├── eval_rollout.py       # autoregressive rollout evaluation
│   └── ...
├── grasp_logs_*/             # grasp logs for 10 objects (15 trial CSVs each)
└── jiezhi.xlsx               # medium-parameter table (read during training)
```

Each trial is a 15-D sequence: left/right finger force, proximity, IMU, and the medium state `s(t)`.

## Setup

```bash
pip install -r cm_tfnet/requirements.txt
```

Dependencies: `torch>=2.0`, `numpy`, `pandas`, `openpyxl`, `matplotlib`.

Run the commands below from the repository root.

## Training

Full CM-TFNet (trial split; predict 1 s after water exit):

```bash
python -m cm_tfnet.train_rollout --model cm_tfnet --split-mode trial --exit-extra 1
```

### Ablations

| Flag | Description |
|------|-------------|
| `no_gate` / `a1` | Mean fusion, no gating |
| `no_multimodal` / `a2` | Single 15-D encoder |
| `no_backbone` / `a3` | Remove TemporalBackbone |
| `no_med` / `a4` | Remove `s(t)` |
| `no_prox` / `a5a` | Remove proximity |
| `no_imu` / `a5b` | Remove IMU |

```bash
python -m cm_tfnet.train_rollout --model no_gate --split-mode trial --exit-extra 1
```

### Baselines

```bash
python -m cm_tfnet.train_rollout --model lstm --match-capacity --split-mode trial --exit-extra 1
python -m cm_tfnet.train_rollout --model gru --match-capacity --split-mode trial --exit-extra 1
python -m cm_tfnet.train_rollout --model tcn --match-capacity --split-mode trial --exit-extra 1
```

`--match-capacity` scales each baseline to roughly the same parameter count as CM-TFNet (~377k).

Checkpoints are written to `checkpoints/` by default. That directory is gitignored and created automatically during training.

## Evaluation

```bash
python -m cm_tfnet.eval_rollout --ckpt checkpoints/best.pt
```

## Dataset

| Directory | Object |
|-----------|--------|
| `grasp_logs_beike` | Shell |
| `grasp_logs_mosha` | Frosted |
| `grasp_logs_mosilian` | Mosilian bottle |
| `grasp_logs_paomo` | Foam |
| `grasp_logs_qiu` | Silicone ball |
| `grasp_logs_ruanzhu` | Soft cylinder |
| `grasp_logs_suliaoping` | Plastic bottle |
| `grasp_logs_taocibei` | Ceramic cup |
| `grasp_logs_touming` | Transparent |
| `grasp_logs_yingzhu` | Rigid cylinder |
