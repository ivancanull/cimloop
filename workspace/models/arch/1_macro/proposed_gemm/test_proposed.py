import sys
import os
import joblib


THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..', '..', '..', '..')))
from scripts import utils as utl

stats = utl.quick_run(MACRO_NAME, dnn='gpt2_medium', layer='002', max_utilization=False)
print(stats)

# stats = joblib.Parallel(n_jobs=None)(
#             joblib.delayed(utl.quick_run)(
#                 MACRO_NAME,
#                 dnn='alexnet', 
#                 layer='0',
#                 max_utilization=False
#             )
# )

# for r in stats:
#     print(r.tops_per_w)