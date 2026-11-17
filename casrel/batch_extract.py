import os
import json
import torch
from transformers import BertTokenizerFast
from model import SPOModel

# 配置
MODEL_PATH = "./checkpoint/best.pt"
REL2ID_PATH = "./data/ultra_high_building/rel2id.json"
PRETRAINED_MODEL_PATH = "./models/bert-base-chinese"
INPUT_DIR = "./input_txts"
OUTPUT_DIR = "./output_spo"

# 1. 加载关系、模型、Tokenizer
with open(REL2ID_PATH, 'r', encoding='utf-8') as f:
    rel2id = json.load(f)
id2rel = [None] * len(rel2id)
for r, i in rel2id.items():
    id2rel[i] = r

tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED_MODEL_PATH)
model = SPOModel(PRETRAINED_MODEL_PATH, REL2ID_PATH)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# 抽取函数
def extract_spo(text, sub_th=0.15, obj_th=0.1):
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'][0])
    with torch.no_grad():
        sub_logits, _ = model(enc['input_ids'], enc['attention_mask'])
    start_s = torch.sigmoid(sub_logits[0,:,0])
    end_s   = torch.sigmoid(sub_logits[0,:,1])
    # 主语 span
    spans = [(i, j)
             for i, s in enumerate(start_s) if s>sub_th
             for j in range(i, min(i+10, len(end_s))) if end_s[j]>sub_th]
    spo = []
    for s, e in spans:
        subject = "".join([tokens[x].lstrip("##") for x in range(s,e+1)])
        # 带上主语位置
        with torch.no_grad():
            _, obj_logits = model(enc['input_ids'], enc['attention_mask'], subject_positions=[(s,e)])
        for ridx in range(obj_logits.shape[1]):
            st = torch.sigmoid(obj_logits[0,ridx,:,0])
            et = torch.sigmoid(obj_logits[0,ridx,:,1])
            # 宾语 span
            for i, ss in enumerate(st):
                if ss>obj_th:
                    for j in range(i, min(i+15, len(et))):
                        if et[j]>obj_th:
                            obj = "".join([tokens[x].lstrip("##") for x in range(i, j+1)])
                            spo.append((subject, id2rel[ridx], obj))
                            break
                    break
    return spo

# 确保输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 批量处理
for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".txt"):
        continue
    in_path = os.path.join(INPUT_DIR, fname)
    text = open(in_path, 'r', encoding='utf-8').read()
    spo = extract_spo(text)
    out_path = os.path.join(OUTPUT_DIR, fname.replace('.txt', '_spo.json'))
    with open(out_path, 'w', encoding='utf-8') as wf:
        json.dump(spo, wf, ensure_ascii=False, indent=2)
    print(f"Processed {fname}: {len(spo)} triples")

print("批量抽取完成，结果保存在", OUTPUT_DIR)
