from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor

from .rag import llm
from .tool import tools

load_dotenv()
prompt = """
你是一个学术文献问答助手，严格按照以下步骤回答问题：
1. 思考：是否需要使用工具？如果需要，选择合适的工具；如果不需要，直接回答。
2. 行动：调用工具（仅使用提供的工具）获取结果。
3. 观察：查看工具返回结果，确认是否足够回答问题。
4. 回答：基于工具结果或自身知识，给出清晰、准确的回答。

可用工具：{tools}
工具调用格式需遵循LangChain规范。

用户问题：{input}
对话历史：{chat_history}
"""

def build_agent_chain():
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=prompt
    )
    agent_executor = AgentExecutor( # 执行器
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5
    )
    return agent_executor

def agent_qa(input):
    agent_chain = build_agent_chain()
    tool_descriptions = "".join(tool.description for tool in agent_chain.tools)
    response = agent_chain.invoke({
        "input": input,
        "chat_history": [],
        "tools": tool_descriptions
    })
    return response['output']