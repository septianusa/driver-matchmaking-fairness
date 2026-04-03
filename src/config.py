from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_OUTPUT_DIR = BASE_DIR / "data" / "output"

TOTAL_BATCH = 1440
MAX_MATCH_DISTANCE_KM = 4.0
ORDER_MAX_CARRYOVER_BATCH = 5

BFS_MAX_DEPTH = 2
BFS_CANDIDATE_LIMIT = 6

DISCOUNT_FACTOR = 0.90
LEARNING_RATE_ALPHA = 0.10

BETA_0 = -2.5
BETA_1 = 1.0
BETA_2 = 2.0
BETA_3 = 1.5

LAMBDA_SCENARIOS = [0.0, 0.3, 0.6]

ORDER_FILE = DATA_RAW_DIR / "order_dataset_surabaya_1day.csv"
PING_FILE = DATA_RAW_DIR / "ping_dataset_surabaya_1day.csv"
DRIVER_PERF_FILE = DATA_RAW_DIR / "driverPerformance_dataset_surabaya_1day.csv"
GRID_FILE = DATA_RAW_DIR / "h3GridValue_dataset_surabaya_1day.csv"