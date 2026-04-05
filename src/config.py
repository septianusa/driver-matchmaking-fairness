from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_OUTPUT_DIR = BASE_DIR / "data" / "output"

TOTAL_BATCH = 14 * 1440
MAX_MATCH_DISTANCE_KM = 2.0
ORDER_MAX_CARRYOVER_BATCH = 3

BFS_MAX_DEPTH = 2
BFS_CANDIDATE_LIMIT = 10

DISCOUNT_FACTOR = 0.90
LEARNING_RATE_ALPHA = 0.10

BETA_0 = -2.5
BETA_1 = 1.0
BETA_2 = 2.0
BETA_3 = 1.5

LAMBDA_SCENARIOS = [0.0, 0.1, 0.2, 0.3]
GRID_SCENARIOS = [True, False]

ORDER_FILE = DATA_RAW_DIR / "order_dataset_surabaya_14day.csv"
PING_FILE = DATA_RAW_DIR / "ping_dataset_surabaya_14day.csv"
DRIVER_PERF_FILE = DATA_RAW_DIR / "driverPerformance_dataset_surabaya_14day.csv"
GRID_FILE = DATA_RAW_DIR / "h3GridValue_dataset_surabaya_14day.csv"