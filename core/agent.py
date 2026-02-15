from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from .memory import get_session_history
from model.factory import chat_model
from .tool import build_tools
from utils.prompt_loader import load_system_prompt

load_dotenv()
prompt = load_system_prompt()

def build_agent(file_path, session_id):
    tools = build_tools(file_path, session_id)
    agent = create_agent(
        model=chat_model,
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