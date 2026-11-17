import os
import json
# 1. 创建目录
base_dir = "data/ultra_high_building"
os.makedirs(base_dir, exist_ok=True)

# 2. 构造训练数据（去除重复样本，缩进规范）
train_data = [
    {
        "text": "超高层建筑防雷等级应达到第一类防雷标准，且接地电阻不超过1Ω；水平等电位连接每隔20～30米设置一次，接地网埋深约4–5米。",
        "spo_list": [
            {
                "subject": "超高层建筑防雷等级",
                "predicate": ["应达到", "需满足", "要求为"],
                "object": ["第一类防雷标准"]
            },
            {
                "subject": "接地电阻",
                "predicate": ["不大于", "不超过", "≤"],
                "object": ["1Ω", "1 欧姆"]
            },
            {
                "subject": "水平等电位连接",
                "predicate": ["间隔", "每隔", "设置间距为"],
                "object": ["20～30米", "20–30 m", "20至30米"]
            },
            {
                "subject": "接地网",
                "predicate": ["埋深", "埋设深度为", "深度达"],
                "object": ["4–5米", "4～5 m", "约4–5米"]
            }
        ]
    },
    {
        "text": "基坑内爬塔吊选用STL420型，最大起重量达24t（24 吨），安装于核心筒内以提升抗风性能。",
        "spo_list": [
            {
                "subject": "基坑内爬塔吊",
                "predicate": ["选用", "采用", "型号为"],
                "object": ["STL420型", "STL420"]
            },
            {
                "subject": "STL420型爬塔吊",
                "predicate": ["最大起重量", "起重量上限", "额定载荷"],
                "object": ["24t", "24 吨"]
            },
            {
                "subject": "爬塔吊",
                "predicate": ["安装位置", "布置在", "设置在"],
                "object": ["核心筒内", "核心筒内部"]
            },
            {
                "subject": "核心筒内部安装",
                "predicate": ["提升", "增强", "提高"],
                "object": ["抗风性能", "抗风能力"]
            }
        ]
    },
    {
        "text": "深圳某大悬挑钢桁架结构最大悬挑约28.74m，约29米；材料包含Q355B、Q345GJC、Q390GJC与Q460GJC，施工模拟使用Midas Gen软件。",
        "spo_list": [
            {
                "subject": "大悬挑钢桁架结构",
                "predicate": ["最大悬挑", "悬挑长度", "悬挑跨距为"],
                "object": ["28.74m", "约29米"]
            },
            {
                "subject": "桁架结构材料",
                "predicate": ["包含", "由…组成", "采用"],
                "object": ["Q355B、Q345GJC、Q390GJC、Q460GJC"]
            },
            {
                "subject": "施工模拟",
                "predicate": ["使用软件", "借助", "采用"],
                "object": ["Midas Gen"]
            }
        ]
    },
    {
        "text": "施工电梯运行速度为63m/min（每分钟63米），日均运行时间12h（约12小时），系统效率显著提高。",
        "spo_list": [
            {
                "subject": "施工电梯",
                "predicate": ["运行速度为", "速度达", "速率为"],
                "object": ["63m/min", "每分钟63米"]
            },
            {
                "subject": "施工电梯",
                "predicate": ["日均运行时间为", "平均每天运行", "日运行时长"],
                "object": ["12h", "12小时", "约12小时"]
            }
        ]
    },
    {
        "text": "北京平安幸福中心通过“水平合院+垂直叠院”立体化设计，夏季下沉庭院可降温2–3℃，生态共享大厅挑高20 m，叠院跨越8层。",
        "spo_list": [
            {
                "subject": "北京平安幸福中心",
                "predicate": ["采用设计模式", "应用设计理念", "通过…设计"],
                "object": ["水平合院+垂直叠院"]
            },
            {
                "subject": "下沉庭院",
                "predicate": ["夏季可降低", "降温", "可降至"],
                "object": ["2–3℃", "2℃～3℃"]
            },
            {
                "subject": "生态共享大厅",
                "predicate": ["挑高", "层高", "净高为"],
                "object": ["20 m", "20米"]
            },
            {
                "subject": "垂直叠院",
                "predicate": ["跨越", "跨层", "延伸至"],
                "object": ["8层"]
            }
        ]
    }
]

# 3. 保存 train_data.json
train_path = os.path.join(base_dir, "train_data.json")
with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

# 4. 自动生成 rel2id.json（注意展开 predicate 列表）
rel_set = set()
for item in train_data:
    for spo in item["spo_list"]:
        predicates = spo["predicate"]
        if isinstance(predicates, list):
            for rel in predicates:
                rel_set.add(rel)
        else:
            rel_set.add(predicates)

rel2id = {rel: i for i, rel in enumerate(sorted(rel_set))}
rel2id["default"] = len(rel2id)

# 保存 rel2id.json
rel_path = os.path.join(base_dir, "rel2id.json")
with open(rel_path, "w", encoding="utf-8") as f:
    json.dump(rel2id, f, indent=2, ensure_ascii=False)

# 5. dev_data.json 直接复制训练数据
dev_path = os.path.join(base_dir, "dev_data.json")
with open(dev_path, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=2, ensure_ascii=False)

print("✅ 已创建以下文件：")
print(f"📄 {train_path}")
print(f"📄 {rel_path}")
print(f"📄 {dev_path}")