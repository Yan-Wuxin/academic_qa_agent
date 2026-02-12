from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from .rag import rag_qa

def build_rag_tool(file_path, session_id):
    def rag_tool_func(input: str): # 内置函数直接传参
        return rag_qa(
            file_path=file_path,
            input=input,
            session_id=session_id
        )
    tools = tool(
        name_or_callable="PaperRAG",
        runnable=rag_tool_func,
        description="内容检索工具，用于查找目标论文内的相关内容"
    )
    return tools

def build_tools(file_path, session_id):
    rag_tool = build_rag_tool(file_path, session_id)
    search_tool = DuckDuckGoSearchRun(
        name="Search",
        description="联网搜索工具，用于查找文献外的最新信息、学术名词解释等"
    )
    return [search_tool, rag_tool]