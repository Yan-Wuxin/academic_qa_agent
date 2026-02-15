import yaml
from utils.path_load import get_abs_path

def load_prompt_config(config_path: str=get_abs_path('config/prompts.yml'),
                       encoding: str='utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_rag_config(config_path: str=get_abs_path('config/rag.yml'),
                    encoding: str='utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_chroma_config(config_path: str=get_abs_path('config/chroma.yml'),
                       encoding: str='utf-8'):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

prompts_conf = load_prompt_config()
rag_config = load_rag_config()
chroma_conf = load_chroma_config()

if __name__ == "__main__":
    print(prompts_conf["rag_prompt_path"])