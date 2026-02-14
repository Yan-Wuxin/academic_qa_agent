import os

def get_project_root() -> str:
    current_file = os.path.abspath(__file__) # 当前文件
    current_dir = os.path.dirname(current_file) # 当前文件夹
    project_root = os.path.dirname(current_dir) # 项目根目录
    return project_root

def get_abs_path(relative_path: str) -> str:
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)