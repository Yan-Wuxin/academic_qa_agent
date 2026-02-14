from utils.path_load import get_abs_path
from utils.config_handler import prompts_conf
from utils.logger_handler import logger

def load_system_prompt():
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompt]在yaml中没有main_prompt_path配置项]")
        raise e
    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错, {str(e)}")
        raise e

def load_rag_prompt():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompt]在yaml中没有rag_prompt_path配置项")
        raise e
    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompt]解析rag提示词出错，{str(e)}")
        raise e

def load_rewrite_prompt():
    try:
        rewrite_prompt_path = get_abs_path(prompts_conf["rewrite_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rewrite_prompt]在yaml中没有相关配置项")
        raise e
    try:
        return open(rewrite_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rewrite_prompt]解析rewrite提示词出错：{str(e)}")
        raise e

def load_hyde_prompt():
    try:
        hyde_prompt_path = get_abs_path(prompts_conf["hyde_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rewrite_prompt]在yaml中没有相关配置项")
        raise e
    try:
        return open(hyde_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rewrite_prompt]解析rewrite提示词出错：{str(e)}")
        raise e


if __name__ == "__main__":
    print(load_hyde_prompt())