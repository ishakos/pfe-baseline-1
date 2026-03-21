from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"
DATA_PATH = BASE_DIR.parent / "Data" / "iot_dataset_clean.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_COL = "label"
AUX_TYPE_COL = "type"

TEST_SIZE = 0.20
VAL_SIZE = 0.20   # from full dataset
CV_FOLDS = 5
N_ITER_SEARCH = 20
PRIMARY_SCORING = "f1"

# Drop columns that are risky, weak, identifiers, or likely to hurt generalization
DROP_COLUMNS = [
    "label",
    "type",
    "dst_ip",
    "src_port",
    "dst_port",
    "conn_state",
    "service",
    "dns_query",
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rcode",
    "ssl_subject",
    "ssl_issuer",
    "ssl_established",
    "http_uri",
    "http_user_agent",
    "http_orig_mime_types",
    "http_resp_mime_types",
    "http_status_code",
    "weird_addl",
    "weird_name",
    "weird_notice",
]

OPTIONAL_DROP_COLUMNS = []

# Optional: if these survived cleaning and you want to exclude them too
DROP_COLUMNS = [
    "label",
    "type",
    "src_ip",
]