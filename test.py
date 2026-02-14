from core.agent import agent_qa

input="论文中提出多头注意力（Multi-Head Attention）的目的是什么？"
file_path = "./data/1706.03762v7.pdf"
response = agent_qa(file_path, input)
print(response)
