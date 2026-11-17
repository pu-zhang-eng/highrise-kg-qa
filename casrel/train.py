import os, json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from model import SPOModel
# ====== 配置 ======
PRETRAINED = './models/bert-base-chinese'
BASE_DIR   = './data/ultra_high_building'
TRAIN_PATH = os.path.join(BASE_DIR, 'train_data.json')
REL2ID     = os.path.join(BASE_DIR, 'rel2id.json')
SAVE_MODEL = './checkpoint/best.pt'

EPOCHS     = 50
BATCH_SIZE = 8
MAX_LEN    = 128
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====== 1. 展平 train_data.json ======
# 假设原来 train_data_raw.json 里 predicate/object 都是列表
with open(os.path.join(BASE_DIR,'train_data.json'),'r',encoding='utf8') as f:
    raw = json.load(f)
flat = []
for item in raw:
    text = item['text']
    for spo in item['spo_list']:
        subj = spo['subject']
        for pred in (spo['predicate'] if isinstance(spo['predicate'],list) else [spo['predicate']]):
            for obj in (spo['object'] if isinstance(spo['object'],list) else [spo['object']]):
                flat.append({'text':text,
                             'subject':subj,
                             'predicate':pred,
                             'object':obj})
# build rel2id
rels = sorted({s['predicate'] for s in flat})
rel2id = {r:i for i,r in enumerate(rels)}
rel2id['default'] = len(rel2id)
with open(REL2ID,'w',encoding='utf8') as f: json.dump(rel2id,f,ensure_ascii=False,indent=2)
# group back into spo_list per text
grouped = {}
for spo in flat:
    grouped.setdefault(spo['text'],[]).append(
      {'subject':spo['subject'],'predicate':spo['predicate'],'object':spo['object']})
train_data = [{'text':t,'spo_list':v} for t,v in grouped.items()]
with open(TRAIN_PATH,'w',encoding='utf8') as f: json.dump(train_data,f,ensure_ascii=False,indent=2)

# ====== 2. Dataset & Encode ======
tokenizer = BertTokenizerFast.from_pretrained(PRETRAINED)
class SPOTrainDataset(Dataset):
    def __init__(self, data_path, rel2id):
        self.data = json.load(open(data_path,'r',encoding='utf8'))
        self.rel2id = rel2id
    def __len__(self): return len(self.data)
    def __getitem__(self,idx):
        x = self.data[idx]
        text, spo = x['text'], x['spo_list']
        enc = tokenizer(text, max_length=MAX_LEN, padding='max_length',
                        truncation=True, return_offsets_mapping=True)
        input_ids = torch.tensor(enc['input_ids'])
        attn_mask = torch.tensor(enc['attention_mask'])
        offsets   = enc['offset_mapping']
        # offsets-> token_positions
        tpos = [None if s==e==0 else (s,e) for s,e in offsets]
        # subject labels
        subj_lbl = torch.zeros(MAX_LEN,2)
        # object labels
        num_rel = len(self.rel2id)-1
        obj_lbl  = torch.zeros(num_rel, MAX_LEN,2)
        sub_pos  = None
        # mark subject & object
        for spo_item in spo:
            s, p, o = spo_item['subject'], spo_item['predicate'], spo_item['object']
            si = text.find(s); ei = si+len(s)
            # token index
            st = next((i for i,pos in enumerate(tpos) if pos and pos[0]<=si<pos[1]),-1)
            en = next((i for i,pos in enumerate(tpos) if pos and pos[0]< ei<=pos[1]),-1)
            if st>=0 and en>=0:
                subj_lbl[st,0]=1; subj_lbl[en,1]=1
                sub_pos = (st,en)
                rid = self.rel2id.get(p,None)
                oi = text.find(o)
                oj = oi+len(o)
                ot = next((i for i,pos in enumerate(tpos) if pos and pos[0]<=oi<pos[1]),-1)
                oe = next((i for i,pos in enumerate(tpos) if pos and pos[0]< oj<=pos[1]),-1)
                if rid is not None and rid< num_rel and ot>=0 and oe>=0:
                    obj_lbl[rid,ot,0]=1
                    obj_lbl[rid,oe,1]=1
        if not sub_pos:
            sub_pos=(0,0)
        return {
          'input_ids':input_ids,'attention_mask':attn_mask,
          'subject_labels':subj_lbl,'object_labels':obj_lbl,
          'subject_start':sub_pos[0],'subject_end':sub_pos[1]
        }

# ====== 3. Training Loop ======
dataset = SPOTrainDataset(TRAIN_PATH,rel2id)
loader  = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
model   = SPOModel(PRETRAINED,REL2ID).to(DEVICE)
optim   = torch.optim.Adam(model.parameters(),lr=2e-5)
bce     = torch.nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    total_loss=0
    model.train()
    for batch in loader:
        ids = batch['input_ids'].to(DEVICE)
        mask= batch['attention_mask'].to(DEVICE)
        sb = batch['subject_labels'].to(DEVICE)
        ob = batch['object_labels'].to(DEVICE)
        ss = batch['subject_start'].tolist()
        se = batch['subject_end'].tolist()
        subpos = list(zip(ss,se))
        optim.zero_grad()
        sub_logits, obj_logits = model(ids,mask,subject_positions=subpos)
        # subj loss
        ls = bce(sub_logits[:,:,0], sb[:,:,0]) + bce(sub_logits[:,:,1], sb[:,:,1])
        # obj loss
        lo = bce(obj_logits[:,:,:,0], ob[:,:,:,0]) + bce(obj_logits[:,:,:,1], ob[:,:,:,1])
        loss = (ls+lo)/2
        loss.backward(); optim.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} loss {total_loss/len(loader):.4f}")

# 保存
torch.save(model.state_dict(), SAVE_MODEL)
print("训练完毕，模型保存到", SAVE_MODEL)