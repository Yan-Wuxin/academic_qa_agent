from model.factory import ChatModelFactory

class AgentBuilder:
    @staticmethod
    def build_agent():
        llm = ChatModelFactory().generator()
