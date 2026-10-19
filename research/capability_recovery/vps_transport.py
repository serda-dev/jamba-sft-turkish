"""Load the deployment-only VPS helper on use, not during CPU module import."""
import importlib
from pathlib import Path
import sys


class VpsTransport:
    def __init__(self, root):
        self.root = root
        self.KEY = Path('/home/serda/.config/vastai/linguai_team_api_key')

    def __getattr__(self, name):
        sys.path.insert(0, '/home/serda/cpt-production')
        try:
            helper = importlib.import_module('scripts.cpt_vps_supervisor')
        except ModuleNotFoundError as error:
            raise RuntimeError('VPS deployment helper required for live rental/SSH operations') from error
        helper.ROOT = self.root
        return getattr(helper, name)
