import array
import sys
import os

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MACRO_NAME = os.path.basename(THIS_SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(THIS_SCRIPT_DIR, '..', '..', '..', '..')))
from scripts import utils as utl
import joblib



def run_set_variables(variables, array_rows=32, array_cols=32):
    # Load in the specification
    spec = utl.get_spec("proposed", 
                        system="ws_dummy_buffer_one_macro",
                        max_utilization=True,
                        dnn='alexnet', layer='0')

    # Enable the MAX_UTILIZATION variable. This will generate a
    # workload that maximizes the utilization of the array.
    spec.variables["MAX_UTILIZATION"] = True

    # Set the array size
    spec.architecture.find("row").spatial.meshY = array_rows
    spec.architecture.find("column").spatial.meshX = array_cols
    

    # Set variables
    spec.variables.update(variables)

    return utl.run_mapper(spec)

def main():

    
    results = joblib.Parallel(n_jobs=None)(
        joblib.delayed(utl.quick_run)(
            MACRO_NAME,
            dnn='alexnet', 
            layer='0',
            variables = {
                "N_COLUMNS": n[0],
                "N_ROWS": n[1]
            }
            
        )
        for n in [(16, 16)]
    )

    for r in results:
        print(r.tops_per_w)

if __name__ == "__main__":
    main()