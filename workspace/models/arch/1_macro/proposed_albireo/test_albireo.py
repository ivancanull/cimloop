import sys
import os

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..', '..', '..', '..')))
from scripts import utils as utl

stats = utl.quick_run(MACRO_NAME, dnn='alexnet', layer='0', max_utilization=False)
print(stats.tops_per_w)