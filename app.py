import os
import streamlit as st
from core.rag import rag_qa
from core.memory import get_session_history

st.set_page_config(
    page_title="学术文献问答助手",
    page_icon="📚",
    layout="wide",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = "user_" + str(hash(os.urandom(16))) # random unique id
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("学术文献问答助手")
st.divider()

with st.sidebar: # 侧边栏，上传pdf文件
    st.subheader("上传文献")
    uploaded_file = st.file_uploader("选择PDF论文", type="PDF")
    if uploaded_file is not None:
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"文件上传成功：{uploaded_file.name}")

    if st.button("清空对话记录", type="secondary"):
        history = get_session_history(st.session_state.session_id)
        history.clear()
        st.session_state.chat_history = []
        st.success("已清空")

st.subheader("问答区")
input = st.text_input("请输入您关于这篇论文的问题")
# 纯RAG回答 + agent使用工具回答
col1, col2 = st.columns(2)
with col1:
    rag_btn = st.button("纯文献回答（RAG）", type="primary")
with col2:
    agent_btn = st.button("智能回答（agent + 工具）")

if rag_btn and uploaded_file and input:
    with st.spinner("正在检索文献并组织答案..."):
        try:
            response = rag_qa(file_path, input)
            st.write(response)
        except Exception as e:
            st.error(f"出错了：{str(e)}")

if agent_btn and uploaded_file and input:
    with st.spinner("正在思考（或调用工具）并组织答案..."):
        try:
            response = agent_qa(file_path, input)
            st.write(response)
        except Exception as e:
            st.error(f"出错了：{str(e)}")