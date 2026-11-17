import os, json, jieba.posseg as pseg, torch
from transformers import BertTokenizerFast
from model import SPOModel
# ====== 配置 ======
PRETRAINED    = './models/bert-base-chinese'
REL2ID        = './data/ultra_high_building/rel2id.json'
MODEL_PATH    = './checkpoint/best.pt'
INPUT_DIR     = './input_txts'
OUTPUT_DIR    = './output_spo'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 加载 ======
tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED)
model     = SPOModel(PRETRAINED,REL2ID)
model.load_state_dict(torch.load(MODEL_PATH,map_location='cpu'))
model.eval()

# id2rel
rel2id = json.load(open(REL2ID,'r',encoding='utf8'))
id2rel = {v:k for k,v in rel2id.items()}

def extract_spo(text):
    # 1) jieba posseg 预筛实体候选
    words = list(pseg.cut(text))
    candidates = []
    for i,(w,flag) in enumerate(words):
        if flag.startswith('n') or flag in ('t','m','x'):
            candidates.append((w, i))
    # 转成 token-level spans
    token_spans = []
    tokens = tokenizer(text, return_offsets_mapping=True, truncation=True,
                       max_length=512)
    offset = tokens['offset_mapping']
    for w,wi in candidates:
        st = text.find(w)
        ed = st+len(w)
        ti = next((i for i,(s,e) in enumerate(offset) if s<=st<e),None)
        tj = next((i for i,(s,e) in enumerate(offset) if s< ed<=e),None)
        if ti is not None and tj is not None:
            token_spans.append((ti,tj))
    token_spans = list(set(token_spans))

    spo_list = []
    input_ids = torch.tensor([tokens['input_ids']])
    mask      = torch.tensor([tokens['attention_mask']])
    for (s,e) in token_spans:
        sub_ids = [(s,e)]
        with torch.no_grad():
            _, obj_logits = model(input_ids,mask,subject_positions=sub_ids)
            start_scores = torch.sigmoid(obj_logits[0,:,:,0])
            end_scores   = torch.sigmoid(obj_logits[0,:,:,1])
        for rid in range(start_scores.size(0)):
            rel = id2rel.get(rid)
            # 只保留分数最高的 top-k
            ss = start_scores[rid]
            es = end_scores[rid]
            topk = torch.topk(ss, k=2)
            for j in topk.indices.tolist():
                if ss[j]>0.3:
                    # 找对应 end
                    for k in range(j, min(j+10, es.size(0))):
                        if es[k]>0.3:
                            subj = tokenizer.convert_tokens_to_string(
                                tokenizer.convert_ids_to_tokens(input_ids[0,s:e+1]))
                            objt = tokenizer.convert_tokens_to_string(
                                tokenizer.convert_ids_to_tokens(input_ids[0,j:k+1]))
                            spo_list.append((subj,rel,objt))
                            break
    # 去重
    return list(set(spo_list))

def main():
    for fn in os.listdir(INPUT_DIR):
        if not fn.endswith('.txt'): continue
        path = os.path.join(INPUT_DIR,fn)
        text = open(path,'r',encoding='utf8').read()
        spo  = extract_spo(text)
        out  = os.path.join(OUTPUT_DIR, fn.replace('.txt','_spo.json'))
        json.dump(spo, open(out,'w',encoding='utf8'), ensure_ascii=False, indent=2)
        print(f'✔ {fn} → {len(spo)} 条，保存在 {out}')

if __name__=='__main__':
    main()