from core.rag import rag_qa
from core.memory import get_session_history
from core.agent import agent_qa

input="文章作者分别有谁？"
file_path = "./data/2210.03629v3.pdf"
response = agent_qa(file_path, input)
print(response)