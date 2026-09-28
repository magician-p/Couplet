from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent # 项目根目录

# 数据根目录
DATA_ROOT = PROJECT_ROOT / "data"

# 原始数据目录
RAW_DATA_DIR = DATA_ROOT / "raw"
# 原始数据名称
RAW_IN_DATA_PATH = RAW_DATA_DIR / "in.txt"
RAW_OUT_DATA_PATH = RAW_DATA_DIR / "out.txt"

# 处理后的数据目录
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test"

# 模型保存目录
MODEL_DIR = PROJECT_ROOT / "model"
# 保存模型参数的文件名
MODEL_PARAMS_FILE = MODEL_DIR / "model_params.pkl"

# 使用的预训练模型
PRETRAINED_MODEL_NAME = "OpenMOSS-Team/bart-base-chinese"

# 分词超参
TOKEN_LENGTH = 64

# 训练超参
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-4