from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT
JIEZHI_XLSX = ROOT / "jiezhi.xlsx"

# Physical parameters
WATER_DEPTH_M = 0.085  # 8.5 cm
SOFT_VEL = 0.03  # m/s
SOFT_DIST = 0.03  # m
SOFT_DURATION = SOFT_DIST / SOFT_VEL  # 1.0 s
CRUISE_VEL = 0.1  # m/s
PEAK_TO_LIFT_DELAY = 1.0  # s
# Extra seconds after full water exit; 0 = stop at exit, 1 = predict 1 s more
EXIT_EXTRA_S = 0.0

# Sequence
# Window may include real observations before t0 as context (same as online rollout)
WINDOW = 64
# Lift-to-exit segment is short; stride=1 densifies supervision
STRIDE = 1

# Model
D_MODEL = 64
GRU_HIDDEN = 64
DROPOUT = 0.1

# Training
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 80
SEED = 42
CROSS_MEDIUM_LOSS_WEIGHT = 3.0
SS_START = 0.2
SS_END = 0.85
LAMBDA_SMOOTH = 0.05
UNROLL_STEPS = 12

CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Object split 6:2:2 (remaining objects are train)
VAL_OBJECTS = ("grasp_logs_mosha", "grasp_logs_mosilian")
TEST_OBJECTS = ("grasp_logs_qiu", "grasp_logs_taocibei")
