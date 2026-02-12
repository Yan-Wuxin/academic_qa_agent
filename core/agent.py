from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from .memory import get_session_history
from .rag import llm
from .tool import build_tools

load_dotenv()
prompt = """
你是一个学术文献问答助手，请严格遵循以下工作策略，尽可能准确地回答用户问题：

1. 如果问题涉及论文内容，必须优先调用 PaperRAG 工具获取信息。
2. 如果需要论文范围之外的信息，应调用 Search 工具。
3. 如果一个问题需要多个步骤解决，可以连续调用多个工具。
4. 如果工具返回信息不足，应继续尝试其他工具。
5. 尽量基于工具结果回答问题，而不是猜测。

请始终优先考虑使用工具。

用户问题：{input}

对话历史：{chat_history}
"""

def build_agent(file_path, session_id):
    tools = build_tools(file_path, session_id)
    agent = create_agent(
        model=llm,
        tools=tools, # 工具描述已自动注入系统prompt
        system_prompt=prompt
    )
    return agent

def agent_qa(file_path, input, session_id="default_session"):
    agent = build_agent(file_path, session_id)
    history = get_session_history(session_id)
    messages = history.messages + [HumanMessage(content=input)]
    response = agent.invoke({
        "messages": messages
    })
    history.add_user_message(input) # 官方API供添加agent记忆
    history.add_ai_message(response["messages"][-1].content)
    return response["messages"][-1].content