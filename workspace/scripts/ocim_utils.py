import os
import yaml

from matplotlib.pylab import f
import utils
import pandas as pd
import timeloopfe.v4 as tl

def fetch_all_layers(dnn_name: str):
    dnn_dir = utils.path_from_model_dir(f"workloads/{dnn_name}")
    layer_paths = [
        os.path.join(dnn_dir, l) for l in os.listdir(dnn_dir) if l.endswith(".yaml")
    ]
    return layer_paths

def handle_all_results(stats: tl.OutputStats, df: pd.DataFrame, index: str):
    gr_df = handle_general_results(stats, df, index)
    cr_df = handle_per_component_results(stats, df, index)
    cc_df = pd.concat([gr_df, cr_df], axis=1)
    df = pd.concat([df, cc_df], axis=0)
    return df

def handle_general_results(stats: tl.OutputStats, df: pd.DataFrame, index: str):
    # write results to a dataframe
    # keys: compute, cycles, cycle_seconds, latency, energy, area, tops, tops_per_mm2, tops_per_w, tops_per_mm2_w
    # add to dataframe
    
    temp_data = {
        'compute': stats.computes,
        'cycles': stats.cycles,
        'cycle_seconds': stats.cycle_seconds,
        'latency': stats.latency,
        'energy': stats.energy,
        'area': stats.area,
        'tops': stats.tops,
        'tops_per_mm2': stats.tops_per_mm2,
        'tops_per_w': stats.tops_per_w,
        'tops_per_mm2_w': stats.tops_per_w / stats.area / 1e6
    }
    temp_df = pd.DataFrame(temp_data, index=[index])
    return temp_df

def handle_per_component_results(stats: tl.OutputStats, df: pd.DataFrame, index: str):
    stats.clear_zero_energies()
    temp_data = {}
    for key in stats.per_component_energy.keys():
        temp_data[f'{key}_energy'] = stats.per_component_energy[key]
        temp_data[f'{key}_area'] = stats.per_component_area[key]
        temp_data[f'{key}_power'] = stats.per_component_energy[key] / stats.cycle_seconds / stats.cycles
    temp_df = pd.DataFrame(temp_data, index=[index])
    return temp_df

def test_different_glbs(macro: str, dnn_name: str, system: str, save_name: str):
    """
    Test different GLBs
    """
    # glb depth
    glb_depth_scales = [8, # 512KB
                        4, # 256KB
                        2, # 128KB
                        1, # 64KB
                        ]
    
    layer_paths = fetch_all_layers(dnn_name)
    stats = utils.parallel_test(
        utils.delayed(utils.quick_run)(
            macro, 
            layer=l, 
            max_utilization=False, 
            system=system,
            variables=dict(GLB_DEPTH_SCALE=scale)
        )
        for l in layer_paths
        for scale in glb_depth_scales
    ) 
    df = pd.DataFrame()
    for idx_l, l in enumerate(layer_paths):
        for idx_scale, scale in enumerate(glb_depth_scales):
            index = f"{macro}_{dnn_name}_{l}_{scale}"
            df = handle_all_results(stats[idx_l * len(glb_depth_scales) + idx_scale], df, index)
    df.to_csv(f"./results/results_{save_name}_{dnn_name}.csv")

def test_full_dnn(macro: str, dnn_name: str, system: str):
    """
    Test full DNN
    """
    # alexnet
    layer_paths = fetch_all_layers(dnn_name)
    stats = utils.parallel_test(
        utils.delayed(utils.quick_run)(
            macro, 
            layer=l, 
            system=system,
            max_utilization=False
        )
        for l in layer_paths
    ) 
    df = pd.DataFrame()
    for l, stat in zip(layer_paths, stats):
        index = f"{macro}_{dnn_name}_{l}"
        df = handle_all_results(stat, df, index)
    os.makedirs(f"results/{dnn_name}", exist_ok=True)
    df.to_csv(f"results/{dnn_name}/results_{macro}.csv")

def load_yaml_to_dict(yaml_path):
    """
    Load a YAML file and return its contents as a dictionary.
    """
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)  # Use safe_load to avoid executing arbitrary code
    return data

def layer_file_to_name(layer_file):
    return layer_file.split('/')[-1].replace('.yaml', '')

def test_hybrid_mapping(macro_regular: str, macro_hybrid: str, mapping_config:str, dnn_name: str, system: str):
    """
    Test hybrid mapping
    """
    # hybrid mapping config
    # load from mapping config yaml
    mapping_dict = load_yaml_to_dict(mapping_config)
    layer_paths = fetch_all_layers(dnn_name)
    
    stats = utils.parallel_test(
        utils.delayed(utils.quick_run)(
            macro_hybrid if mapping_dict[layer_file_to_name(l)] else macro_regular, 
            layer=l, 
            system=system,
            max_utilization=False
        )
        for l in layer_paths
    ) 
    df = pd.DataFrame()
    for l, stat in zip(layer_paths, stats):
        index = f"proposed_mrr_1bit_input_hybrid_{dnn_name}_{l}"
        df = handle_all_results(stat, df, index)
    os.makedirs(f"results/{dnn_name}", exist_ok=True)
    df.to_csv(f"results/{dnn_name}/results_proposed_mrr_1bit_input_hybrid.csv")

def test_two_layers(macro: str, dnn_name: str, system: str, save_name: str):
    """
    Test two layers
    """
    layer_paths = fetch_all_layers(dnn_name)
    stats = utils.parallel_test(
        utils.delayed(utils.quick_run)(
            macro, 
            layer=l, 
            max_utilization=False, 
            system=system
        )
        for l in layer_paths[:2]
    ) 
    df = pd.DataFrame()
    for l, stat in zip(layer_paths[:2], stats):
        index = f"{macro}_{dnn_name}_{l}"
        df = handle_all_results(stat, df, index)
    df.to_csv(f"./results/results_{save_name}_{dnn_name}.csv")

def test_one_layer(macro: str, dnn_name: str, layer: str, system: str, variables=None):
    """
    Test one layer
    """
    layer_paths = fetch_all_layers(dnn_name)
    stats = utils.quick_run(
        macro,
        dnn=dnn_name,
        layer=layer,
        system=system,
        max_utilization=False,
    )

    
    df = pd.DataFrame()
    index = f"{macro}_{dnn_name}_{layer}"
    df = handle_all_results(stats, df, index)
    df.to_csv(f"temp/results_{macro}_{dnn_name}_{layer}.csv")
