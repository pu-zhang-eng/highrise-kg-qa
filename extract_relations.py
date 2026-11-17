import re
import os
import csv
import sys
from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Set
sys.setrecursionlimit(10000)

class DependencyTree:
    def __init__(self, word: str, pos: str, dep: str, head: int, index: int):
        self.word = word
        self.pos = pos
        self.dep = dep
        self.head = head
        self.index = index
        self.children = []

class RelationExtractor:
    def __init__(self):
        self.hanlp = self._initialize_hanlp()
        self.stopwords = self._load_stopwords()
        self.predicate_mappings = self._load_predicate_mappings()

    def _initialize_hanlp(self):
        try:
            import hanlp
            print(f"HanLP版本: {hanlp.__version__}")
            HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
            return HanLP
        except Exception as e:
            print(f"HanLP初始化失败: {e}")
            return None

    def _load_stopwords(self) -> Set[str]:
        stopwords = set()
        try:
            with open('data/stopwords_full.txt', 'r', encoding='utf-8') as f:
                stopwords = {line.strip() for line in f}
            print(f"成功加载 {len(stopwords)} 个停用词")
        except Exception as e:
            print(f"加载停用词失败: {e}")
        return stopwords

    def _load_predicate_mappings(self) -> Dict[str, str]:
        return {
            '建造|建设|修建|兴建|营建': '建造关系',
            '位于|坐落|处于|地处': '位置关系',
            '包含|包括|囊括|设有': '包含关系',
            '设计|规划|布局|营造': '设计关系',
            '装饰|点缀|美化|布置': '装饰关系',
            '改造|整治|修复|重建': '改造关系',
            '管理|负责|运营|维护': '管理关系',
            '借鉴|参考|效仿|仿照': '参考关系',
            '采用|使用|利用|运用': '应用关系'
        }

    def build_dependency_tree(self, words: List[str], postags: List[str], arcs) -> DependencyTree:
        nodes = []
        for i, (word, pos) in enumerate(zip(words, postags)):
            arc = arcs[i]
            if isinstance(arc, dict):
                head = arc.get('head', 0)
                dep_label = arc.get('label', '')
            elif isinstance(arc, (list, tuple)) and len(arc) >= 2:
                head, dep_label = arc[0], arc[1]
            else:
                head, dep_label = 0, ''
            node = DependencyTree(word, pos, dep_label, head, i)
            nodes.append(node)
        root = None
        for node in nodes:
            if node.head == 0:
                root = node
            else:
                parent = nodes[node.head - 1]
                parent.children.append(node)
        if not root and nodes:
            root = nodes[0]
        return root

    def find_main_predicate(self, tree: DependencyTree) -> Optional[DependencyTree]:
        if tree.dep in ['root', 'HED'] and tree.pos.startswith('v'):
            return tree
        if tree.word in ['是', '为'] and tree.pos.startswith('v'):
            return tree
        if tree.pos.startswith('v') and tree.dep in ['dep', 'ccomp', 'rcmod']:
            return tree
        for child in tree.children:
            predicate = self.find_main_predicate(child)
            if predicate:
                return predicate
        return None

    def find_subject_object(self, predicate_node: DependencyTree, tree: DependencyTree) -> Tuple[
        Optional[str], Optional[str]]:
        subject = None
        object_ = None
        for child in predicate_node.children:
            if child.dep in ['nsubj', 'SBV', 'top'] and child.pos.startswith(('n', 'v', 'a')):
                subject = self._get_full_phrase(child)
            elif child.dep in ['dobj', 'VOB', 'attr', 'pobj'] and child.pos.startswith(('n', 'v', 'a')):
                object_ = self._get_full_phrase(child)
        if not subject:
            for child in predicate_node.children:
                if child.dep in ['dep', 'conj'] and child.pos.startswith(('n', 'v', 'a')):
                    subject = self._get_full_phrase(child)
                    break
        if not object_:
            for child in predicate_node.children:
                if child.dep in ['dep', 'conj', 'rcmod'] and child.pos.startswith(('n', 'v', 'a')):
                    object_ = self._get_full_phrase(child)
                    break
        if predicate_node.dep == 'pass':
            subject, object_ = object_, subject
        return subject, object_

    def _get_full_phrase(self, node: DependencyTree) -> str:
        words = []
        for child in node.children:
            if child.dep in ['advmod', 'amod', 'nn', 'assmod', 'rcmod', 'nummod', 'clf'] and child.index < node.index:
                words.extend(self._get_full_phrase(child).split())
        words.append(node.word)
        for child in node.children:
            if child.dep in ['advmod', 'amod', 'nn', 'assmod', 'rcmod', 'nummod', 'clf'] and child.index > node.index:
                words.extend(self._get_full_phrase(child).split())
        return ' '.join(words)

    def normalize_predicate(self, predicate: str) -> str:
        # 健壮性: 判空和类型
        if not predicate or not isinstance(predicate, str):
            return ""
        verb_mappings = {
            '位于|坐落|处于|地处': '位于',
            '分布|散布|遍布': '分布',
            '包含|包括|囊括': '包含',
            '组成|构成|形成': '构成',
            '用于|用作|作为': '用于',
            '具有|拥有|具备': '具有',
            '提供|给予|供给': '提供',
            '设计|规划|布局': '设计',
            '建造|建设|修建|营建': '建造',
            '营造|布置|配置': '营造',
            '应用|采用|使用': '应用',
            '借鉴|参考|效仿': '借鉴',
            '影响|作用|制约': '影响',
            '决定|支配|控制': '决定',
            '促进|推动|带动': '促进',
            '表现|展现|显示': '表现',
            '体现|反映|呈现': '体现',
            '变化|改变|转变': '变化',
            '发展|演变|进化': '发展',
            '分类|划分|区分': '分类',
            '属于|归于|隶属': '属于'
        }
        for patterns, normalized in verb_mappings.items():
            # patterns是'位于|坐落|处于|地处'
            if any(p and isinstance(p, str) and p in predicate for p in patterns.split('|')):
                return normalized
        # 去除部分无意义词和符号，包括 ...、……（省略号）、和中英文标点
        predicate = re.sub(r'着|了|过|的|地|…|\.{3,}', '', predicate)
        return predicate.strip()

    def clean_entity(self, entity: str) -> str:
        if not entity or not isinstance(entity, str):
            return ""
        entity = re.sub(r'[《》，。、；：？！\"\'()（）…\.]', '', entity)
        entity = re.sub(r'\d+\s*[年个篇章节]', '', entity)
        meaningless_words = {'是', '为', '将', '把', '被', '有', '就是', '加之', '使得', '都'}
        words = [w for w in entity.split() if w not in meaningless_words]
        if len(words) > 3:
            try:
                pos_words = [(w, p) for w, p in zip(words, self.hanlp(entity)['pos/pku'])]
                core_words = []
                for word, pos in pos_words:
                    if pos.startswith(('n', 'v')) or len(core_words) < 2:
                        core_words.append(word)
                words = core_words[:3]
            except Exception:
                words = words[:3]
        entity = ''.join(words)
        if not entity and words:
            return ''.join(words[:3])
        return entity

    def is_valid_relation(self, subject: str, relation: str, object_: str) -> bool:
        if not subject or not object_ or not relation:
            return False
        if subject == object_:
            return False
        if len(subject) > 10 or len(object_) > 10:
            return False
        if subject in self.stopwords or object_ in self.stopwords:
            return False
        try:
            subject_pos = self.hanlp(subject)['pos/pku'][0]
            object_pos = self.hanlp(object_)['pos/pku'][0]
            if not (subject_pos.startswith(('n', 'v', 'a')) and object_pos.startswith(('n', 'v', 'a'))):
                return False
        except Exception:
            return False
        return True

    def classify_entity(self, entity: str, pos: str) -> str:
        architecture_terms = {
            '建筑', '大楼', '大厦', '高楼', '高层', '超高层', '摩天楼', '塔', '结构', '框架', '剪力墙',
            '核心筒', '筒体', '基础', '地基', '桩', '幕墙', '钢结构', '混凝土', '钢筋', '电梯',
            '消防', '避难', '停机坪', '空间', '造型', '线条', '色彩', '构图', '设计', '规划', '布局', '施工', '建造', '材料'
        }
        design_terms = {
            '点', '线', '面', '体', '形状', '纹理', '肌理', '质感', '图案', '图形',
            '几何', '抽象', '具象', '对称', '均衡', '韵律', '节奏', '比例', '尺度', '层次', '立面'
        }
        tech_terms = {
            '技术', '方法', '体系', '系统', '模式', '机制', '原理', '理论', '手段', '工具',
            '分割', '组合', '排列', '布置', '配置', '处理', '应用', '实践', '创作', '表现',
            '抗震', '减震', '隔震', '阻尼', '仿真'
        }
        style_terms = {
            '风格', '流派', '主义', '艺术', '美学', '传统', '现代', '古典', '自然', '人文', '中式', '西式'
        }
        org_terms = {
            '大学', '学院', '研究所', '部门', '机构', '单位', '公司', '集团', '协会', '中心',
            '出版社', '设计院', '工作室', '实验室', '学校', '系所', '事务所'
        }
        project_terms = {
            '工程', '项目', '计划', '方案', '建设', '改造', '整治', '保护', '开发', '论文', '研究', '实验', '调研', '考察', '考量'
        }
        if not pos:
            return 'Entity'
        if pos.startswith('n'):
            for term in architecture_terms:
                if term in entity:
                    return 'Architecture'
            for term in design_terms:
                if term in entity:
                    return 'Design'
            for term in tech_terms:
                if term in entity:
                    return 'Technology'
            for term in style_terms:
                if term in entity:
                    return 'Style'
            for term in org_terms:
                if term in entity:
                    return 'Organization'
            for term in project_terms:
                if term in entity:
                    return 'Project'
            return 'Entity'
        elif pos.startswith('v'):
            for term in tech_terms:
                if term in entity:
                    return 'Technology'
            return 'Action'
        elif pos.startswith('a'):
            return 'Property'
        else:
            return 'Entity'

    def extract_relations_from_sentence(self, sentence: str) -> List[Dict]:
        try:
            result = self.hanlp(sentence)
            if isinstance(result, (tuple, list)):
                result = result[0]
            if hasattr(result, "to_dict"):
                result = result.to_dict()
            if not isinstance(result, dict):
                return []
            if 'tok/fine' not in result or 'pos/pku' not in result or 'dep' not in result:
                return []
            words_data = result['tok/fine']
            postags_data = result['pos/pku']
            deps_data = result['dep']
            if not words_data or not postags_data or not deps_data:
                return []
            if isinstance(words_data[0], list):
                words = words_data[0]
            else:
                words = list(words_data)
            if isinstance(postags_data[0], list):
                postags = postags_data[0]
            else:
                postags = list(postags_data)
            if isinstance(deps_data, list) and isinstance(deps_data[0], dict):
                deps = deps_data
            elif isinstance(deps_data, list) and isinstance(deps_data[0], list):
                deps = deps_data[0]
            else:
                deps = deps_data
            if not words or not postags or not deps:
                return []
            tree = self.build_dependency_tree(words, postags, deps)
            predicate = self.find_main_predicate(tree)
            if not predicate:
                return []
            subject, object_ = self.find_subject_object(predicate, tree)
            relations = []
            if subject and object_:
                subject = self.clean_entity(subject)
                object_ = self.clean_entity(object_)
                relation = self.normalize_predicate(predicate.word)
                if self.is_valid_relation(subject, relation, object_):
                    subject_category = self.classify_entity(subject, predicate.pos)
                    object_category = self.classify_entity(object_, predicate.pos)
                    relations.append({
                        'subject': subject,
                        'relation': relation,
                        'object': object_,
                        'subject_category': subject_category,
                        'object_category': object_category,
                        'confidence': 0.8,
                        'method': 'dependency_parsing'
                    })
            return relations
        except Exception as e:
            print(f"句子处理失败: {sentence}")
            print(f"错误: {str(e)}")
            return []

    def extract_relations(self, text: str) -> List[Dict]:
        relations = []
        blocks = text.splitlines()
        for block in blocks:
            block = block.strip()
            if not block or len(block) < 10:
                continue
            # 跳过公式和明显无用内容
            if re.search(r'[=∑∫×\d+\.?\d*[A-Za-z]*]', block) or re.search(r'（\d+）', block) or re.search(r'Q[su]|Q[pd]|β|γ|≥|≤|~', block):
                continue
            if re.match(r'^[A-Za-z\u0370-\u03FF]', block):
                continue
            sentences = re.split(r'[。！？；]', block)
            for sentence in sentences:
                if len(sentence.strip()) < 10:
                    continue
                sentence_relations = self.extract_relations_from_sentence(sentence)
                relations.extend(sentence_relations)
        return relations

    def process_single_file(self, file_path: str) -> Optional[Dict]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            if not text:
                return None
            relations = self.extract_relations(text)
            result = {
                'filename': os.path.basename(file_path),
                'text_length': len(text),
                'relations_count': len(relations),
                'relations': relations
            }
            return result
        except Exception as e:
            print(f"处理文件失败 {os.path.basename(file_path)}: {e}")
            return None

    def process_all_files(self, split_dir: str, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        knowledge_graph_file = os.path.join(output_dir, 'knowledge_graph.csv')
        unique_relations = set()
        all_relations = []
        with open(knowledge_graph_file, 'w', encoding='utf-8', newline='') as f_kg:
            kg_writer = csv.writer(f_kg)
            kg_writer.writerow([
                'Subject', 'Relation', 'Object',
                'Subject_Category', 'Object_Category'
            ])
            txt_files = [f for f in os.listdir(split_dir) if f.endswith('.txt')]
            for filename in tqdm(txt_files, desc="处理文件"):
                file_path = os.path.join(split_dir, filename)
                result = self.process_single_file(file_path)
                if result:
                    for relation in result['relations']:
                        relation_key = (
                            relation['subject'],
                            relation['relation'],
                            relation['object']
                        )
                        if relation_key not in unique_relations:
                            unique_relations.add(relation_key)
                            all_relations.append(relation)
                            kg_writer.writerow([
                                relation['subject'],
                                relation['relation'],
                                relation['object'],
                                relation['subject_category'],
                                relation['object_category']
                            ])
        print(f"\n总共处理了 {len(txt_files)} 个文件")
        print(f"提取出 {len(unique_relations)} 个唯一关系")
        print(f"\n结果已保存到: {knowledge_graph_file}")
        print("\n关系类型分布:")
        relation_counts = defaultdict(int)
        for relation in all_relations:
            relation_counts[relation['relation']] += 1
        for relation, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"- {relation}: {count}个")
        return all_relations

def main():
    print("基于依存句法的超高层建筑知识图谱构建")
    print("=" * 50)
    split_dir = "data/split"
    output_dir = "data/neo4j_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    extractor = RelationExtractor()
    extractor.process_all_files(split_dir, output_dir)
    print("\n处理完成!")

if __name__ == '__main__':
    main()
