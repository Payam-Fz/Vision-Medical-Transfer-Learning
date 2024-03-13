import os
import sys

DEFAULT_JOB_PATH = '/ubc/cs/research/shield/projects/payamfz/medical-ssl-segmentation'
DEFAULT_LOCAL_PATH = '/mnt/samba/research/shield/projects/payamfz/medical-ssl-segmentation'

_CODE_FOLDER_NAME = '/mycode'

def set_path(target_path=None):
    if target_path is None:
        if os.getcwd().endswith('/job'):
            # we are running a job
            target_path = DEFAULT_JOB_PATH
        else:
            # we are running locally
            target_path = DEFAULT_LOCAL_PATH

    if os.getcwd() != target_path:
        print(f"setting cwd to '{target_path}'")
        os.chdir(target_path)
    
    code_path = target_path + _CODE_FOLDER_NAME
    if code_path not in sys.path:
        print(f"adding '{code_path}' to sys.path")
        sys.path.append(code_path)

    root = os.getcwd()
    assert root.endswith('medical-ssl-segmentation'), "Wrong path is set, check the set_path() function"
    os.makedirs(root + '/out', exist_ok=True)
    os.makedirs(root + '/out/job_logs', exist_ok=True)
    # os.makedirs(root + '/out/board', exist_ok=True)
    # os.makedirs(root + '/out/figs', exist_ok=True)
    # os.makedirs(root + '/out/models', exist_ok=True)

def get_proj_path():
    root = os.getcwd()
    assert root.endswith('medical-ssl-segmentation'), "Wrong path is set, check the set_path() function"
    return root

def get_code_path():
    root = os.getcwd()
    assert root.endswith('medical-ssl-segmentation'), "Wrong path is set, check the set_path() function"
    return root + _CODE_FOLDER_NAME

