from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from config import *


def preprocess(data_in_path, data_out_path):
    # 两个文件逐行对应（上联 / 下联）：分别加载后横向拼成两列
    # 注意：不能写成 data_files=[in, out]，那只会纵向拼接成一列，无法区分来源
    ds_in = load_dataset('text', data_files=str(data_in_path), split='train')
    ds_out = load_dataset('text', data_files=str(data_out_path), split='train')
    assert len(ds_in) == len(ds_out), f"行数不一致: {len(ds_in)} vs {len(ds_out)}"

    # 两个 Dataset 的列名都是 text，必须先改名才能横向拼接
    ds_in = ds_in.rename_column('text', 'input')
    ds_out = ds_out.rename_column('text', 'target')
    dataset = concatenate_datasets([ds_in, ds_out], axis=1)

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    def process(x):
        input = [s.replace(' ', '') for s in x['input']]
        target = [s.replace(' ', '') for s in x['target']]
        inputs = tokenizer(text=input, text_target=target, padding='max_length', truncation=True, max_length=TOKEN_LENGTH)
        return inputs
    dataset = dataset.map(process, batched=True, remove_columns=['input', 'target'])
    
    datasetdict=dataset.train_test_split(test_size=0.25)
    datasetdict.save_to_disk(str(PROCESSED_DATA_DIR))


if __name__ == "__main__":
    preprocess(RAW_IN_DATA_PATH, RAW_OUT_DATA_PATH)