import torch
import torch.nn as nn
from transformers import BertModel
import json

class SPOModel(nn.Module):
    def __init__(self, model_path, rel2id_path):
        super(SPOModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size

        # 加载关系映射
        with open(rel2id_path, 'r', encoding='utf-8') as f:
            full_rel2id = json.load(f)
        self.rel2id = {k: v for k, v in full_rel2id.items() if k != "default"}
        self.num_relations = len(self.rel2id)

        # 主语预测层
        self.subject_classifier = nn.Linear(self.hidden_size, 2)

        # 宾语预测层（融入主语信息）
        # 输入维度：BERT隐藏层 + 主语表示 = hidden_size * 2
        self.object_classifiers = nn.ModuleList([
            nn.Linear(self.hidden_size * 2, 2) for _ in range(self.num_relations)
        ])

        # 主语表示投影层
        self.subject_projector = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input_ids, attention_mask, subject_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [B, L, H]

        # 主语预测
        sub_logits = self.subject_classifier(sequence_output)  # [B, L, 2]

        # 初始化宾语logits
        batch_size, seq_len, _ = sequence_output.size()
        obj_logits = torch.zeros((batch_size, self.num_relations, seq_len, 2),
                                 device=sequence_output.device)

        if subject_positions is not None:
            # 生成主语表示
            subject_reps = []
            for i in range(batch_size):
                if i < len(subject_positions):  # 确保索引有效
                    start, end = subject_positions[i]
                    # 平均开始和结束位置的表示
                    start_rep = sequence_output[i, start]
                    end_rep = sequence_output[i, end]
                    subject_rep = (start_rep + end_rep) / 2.0
                else:
                    # 如果没有提供主语位置，使用零向量
                    subject_rep = torch.zeros(self.hidden_size, device=sequence_output.device)

                subject_rep = self.subject_projector(subject_rep)  # 投影
                subject_reps.append(subject_rep)

            subject_reps = torch.stack(subject_reps)  # [B, H]
            # 扩展维度以匹配序列长度 [B, L, H]
            subject_reps = subject_reps.unsqueeze(1).repeat(1, seq_len, 1)

            # 融合主语和上下文信息
            combined_rep = torch.cat([sequence_output, subject_reps], dim=-1)  # [B, L, 2H]

            # 为每个关系预测宾语
            for rel_idx in range(self.num_relations):
                obj_logits[:, rel_idx] = self.object_classifiers[rel_idx](combined_rep)

        return sub_logits, obj_logits