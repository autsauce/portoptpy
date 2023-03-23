from .client import PortfolioOptimizer
import subprocess

def get_git_version():
    try:
        version = subprocess.check_output(['git', 'describe', '--tags']).decode().strip().split('v')[1]
    except subprocess.CalledProcessError:
        version = '0.0.0'

    return version

__all__ = ['PortfolioOptimizer']
__version__ = get_git_version()