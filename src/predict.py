import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import *


def predict(sentence, model, tokenizer):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    sentence = sentence.replace(' ', '')
    inputs = tokenizer(sentence, padding='max_length', truncation=True, max_length=TOKEN_LENGTH, return_tensors='pt', return_token_type_ids=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=21)
        output_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_sentence


if __name__ == "__main__":
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    while True:
        sentence = input('请输入上联: ')
        if sentence == 'exit':
            break
        couplet = predict(sentence, model, tokenizer)
        print(f'下联: {couplet}')