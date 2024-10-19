__all__ = [
    "calc_fnc",
    "checkpoint",
    "errors",
    "metrics",
    "profile",
    "scripts",
    "stats_utils",
    "grab",
    "format_time",
    "cpuinfo",
]

from . import calc_fnc
from . import checkpoint
from . import errors
from . import metrics
from . import profile
from . import scripts
from . import stats_utils
from . import cpuinfo


def grab(var):
    return var.detach().cpu().numpy()


def format_time(s):
    isec = int(s)
    secs = isec % 60
    mins = isec // 60
    hours = mins // 60
    mins %= 60

    return f"{hours:02d}:{mins:02d}:{secs:02d}"
