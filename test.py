from core.agent import agent_qa

input="论文中编码器（Encoder）和解码器（Decoder）分别由哪些组件构成？"
file_path = "./data/1706.03762v7.pdf"
response = agent_qa(file_path, input)
print(response)
