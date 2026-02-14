from core.agent import agent_qa

input="该提纲中要求掌握关于Z变换的哪些知识点？"
file_path = "./data/"
response = agent_qa(file_path, input)
print(response)
