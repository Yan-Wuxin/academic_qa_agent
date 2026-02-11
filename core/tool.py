from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun(
    name="Search",
    description="联网搜索工具，可用于查找文献外的最新信息、学术名词解释等"
)
# calculator_tool = load_tools(["llm-math"], llm=llm)

tools = [search_tool]

# tools_description = """
# 1. Search：联网搜索工具，可用于查找文献外的最新信息、学术名词解释等；
# 2. Calculator：计算器工具，用于计算文献中的数值、公式等。
# """