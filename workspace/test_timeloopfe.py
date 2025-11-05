import timeloopfe.v4 as tl
from joblib import Parallel, delayed
from scripts import *
from scripts.notebook_utils import *


ARCH_PATH = "./models/arch/1_macro/basic_analog/arch.yaml"

def run_basic_analog_spec(array_rows: int, array_columns: int):
    # Load in the specification
    spec = get_spec("albireo_isca_2021")

    # Enable the MAX_UTILIZATION variable. This will generate a
    # workload that maximizes the utilization of the array.
    spec.variables["MAX_UTILIZATION"] = True

    # Set the number of rows and columns in the array
    spec.architecture.find("column").spatial.meshX = array_columns
    spec.architecture.find("row").spatial.meshY = array_rows

    return run_mapper(spec)


results = run_basic_analog_spec(array_rows=16, array_columns=16)

exit()

# Basic setup. Gathers input files, checks for errors
spec = tl.Specification.from_yaml_files(
  "/home/zhanghf/projects/parsed-processed-input.yaml"
)
# Call Timeloop mapper
tl.call_mapper(spec, output_dir="/home/zhanghf/projects")
# Call Accelergy verbose
tl.call_accelergy_verbose(spec, output_dir="/home/zhanghf/projects")
 
# # Multiprocessed design space exploration
# def run_mapper_with_spec(buf_size: int):
#   spec = tl.Specification.from_yaml_files(
#     "your_input_file.yaml", "your_other_input_file.yaml"
#   )
#   spec.architecture.find("my_buffer").attributes.depth = buf_size
#   return tl.call_mapper(spec, output_dir=f"outputs_bufsize={buf_size}")
 
# buf_sizes = [1024, 2048, 4096, 8192, 16384]
# results = Parallel(n_jobs=8)(
#   delayed(run_mapper_with_spec)(buf_size) for buf_size in buf_sizes
# )