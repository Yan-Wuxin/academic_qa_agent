import os
import streamlit as st
from core.rag import rag_qa
from core.memory import get_session_history
from core.agent import agent_qa

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
        st.session_state.file_path = file_path # 防止页面刷新丢失
        st.success(f"文件上传成功：{uploaded_file.name}")

    if st.button("清空对话记录", type="secondary"):
        history = get_session_history(st.session_state.session_id)
        history.clear()
        st.session_state.chat_history = []
        st.success("已清空")

st.subheader("对话历史")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

st.subheader("问答区")
input = st.text_input("请输入您关于这篇文章的问题")

# 纯RAG回答 + agent使用工具回答
col, = st.columns(1) # 返回数组
with col:
    agent_btn = st.button("发送", type="primary")

if agent_btn and uploaded_file and input:
    st.session_state.chat_history.append({"role": "user", "content": input})
    with st.spinner("正在思考并组织答案..."):
        try:
            response = agent_qa(
                file_path=st.session_state.file_path,
                input=input,
                session_id=st.session_state.session_id,
            )
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
        except Exception as e:
            st.error(f"出错了：{str(e)}")