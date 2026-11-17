import streamlit as st
import os
from py2neo import Graph
from PIL import Image
# ------- 自定义美化CSS ---------
st.set_page_config(page_title="超高层建筑知识图谱问答系统by张普", page_icon="🏗️", layout="centered")

custom_css = """
<style>
body {
    background-color: #ecf4fc !important;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stButton>button {
    color: white;
    background: linear-gradient(90deg, #007fff 20%, #00b7ff 100%);
    border-radius: 8px;
    height: 3em;
    font-weight: bold;
    border: none;
}
.stTextInput>div>div>input {
    border: 2px solid #00b7ff;
    border-radius: 8px;
}
.stChatMessage {
    border-radius: 8px;
    background: #f3f7fa;
    padding: 10px 15px;
    margin-bottom: 10px;
}
.answer-block {
    background: #e6fffc;
    border-left: 5px solid #f9d923;
    padding: 16px;
    border-radius: 8px;
    font-size: 18px;
    margin: 15px 0;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ------- 顶部Logo和标题 -------
logo_path = "logo.png"  # 放一张图片在同目录下，比如“logo.png”
if os.path.exists(logo_path):
    st.image(logo_path, width=120)
st.markdown("<h2 style='color:#007fff;font-weight:bold;'>建筑知识图谱问答系统</h2>", unsafe_allow_html=True)
st.markdown('<hr style="border:1.5px solid #00b7ff;">', unsafe_allow_html=True)

# ------- 连接Neo4j数据库（地址和账号密码从环境变量读取，不写在代码里） ---------
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    st.error("请先设置环境变量 NEO4J_PASSWORD（见 README）")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_graph():
    return Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

graph = get_graph()

# ------- 问答函数 -----------
def answer_question(question):
    # 用户输入通过 $参数 传给 Cypher，不直接拼进查询字符串（防止 Cypher 注入）
    # 问“谁满足了X”（要放在“X满足了什么”前面，否则含“满足”的问题会先被下面那个分支接走）
    if question.startswith("谁") or ("谁" in question and "满足" in question):
        obj = question.split("满足")[-1].replace("？", "").replace("?", "")
        cypher = '''
        MATCH (s)-[r:满足]->(o)
        WHERE o.name = $obj
        RETURN s.name as answer
        '''
        result = graph.run(cypher, obj=obj).data()
        if result:
            return f"满足{obj}的是：{'，'.join(row['answer'] for row in result)}"
        else:
            return "没有找到答案"

    # 问“X满足了什么”
    if "满足" in question:
        subject = question.split("满足")[0]
        cypher = '''
        MATCH (s)-[r:满足]->(o)
        WHERE s.name = $subject
        RETURN o.name as answer
        '''
        result = graph.run(cypher, subject=subject).data()
        if result:
            return f"{subject}满足：{'，'.join(row['answer'] for row in result)}"
        else:
            return "没有找到答案"

    # 问“X是什么”
    if "是什么" in question:
        subject = question.replace("是什么", "").replace("？", "").replace("?", "")
        cypher = '''
        MATCH (s)-[r]->(o)
        WHERE s.name = $subject AND type(r) = '是'
        RETURN o.name as answer
        '''
        result = graph.run(cypher, subject=subject).data()
        if result:
            return f"{subject}是：{'，'.join(row['answer'] for row in result)}"
        else:
            return "没有找到答案"

    return "暂时无法理解你的问题"

# ------- 历史问答会话 -------
if 'history' not in st.session_state:
    st.session_state['history'] = []

with st.form("qa_form"):
    user_question = st.text_input("请输入你的问题：", value="", placeholder="例如：静载试验是什么、承载能力满足了什么...")
    submitted = st.form_submit_button("提交查询")
    if submitted and user_question.strip():
        answer = answer_question(user_question.strip())
        st.session_state['history'].append((user_question, answer))

# 显示历史问答
for q, a in reversed(st.session_state['history']):
    st.markdown(f"**Q：{q}**")
    st.markdown(f'<div class="answer-block">A：{a}</div>', unsafe_allow_html=True)
