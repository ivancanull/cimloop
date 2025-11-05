# Basic optical cim script


from matplotlib.pylab import f
import utils
import pandas as pd
import timeloopfe.v4 as tl





def handle_results(stats, df: pd.DataFrame, index: str):
    # write results to a dataframe
    df.loc[index, 'compute'] = stats.computes
    df.loc[index, 'cycles'] = stats.cycles
    df.loc[index, 'cycle_seconds'] = stats.cycle_seconds
    df.loc[index, 'latency'] = stats.latency
    df.loc[index, 'energy'] = stats.energy
    df.loc[index, 'area'] = stats.area
    df.loc[index, 'tops'] = stats.tops
    df.loc[index, 'tops_per_mm2'] = stats.tops_per_mm2
    df.loc[index, 'tops_per_w'] = stats.tops_per_w
    df.loc[index, 'tops_per_mm2_w'] = stats.tops_per_w / stats.area / 1e6
    return df

def handle_energy_area_power(stats: tl.OutputStats, df: pd.DataFrame, index: str):
    stats.clear_zero_energies()
    for key in stats.per_component_energy.keys():
        df.loc[index, f'{key}_energy'] = stats.per_component_energy[key]
        df.loc[index, f'{key}_area'] = stats.per_component_energy[key]
        df.loc[index, f'{key}_power'] = stats.per_component_energy[key] / stats.cycle_seconds / stats.cycles
    return df

def handle_results_proposed(stats: tl.OutputStats, df: pd.DataFrame, index: str, wo_dl: bool = False):
    # handle results for proposed architecture

    stats.combine_per_component_energy(
        ["input_mrr", "input_mrr_wdm"], "Input_DAC"
    )
    stats.combine_per_component_energy(
        ["output_adc", "photodiode_output_readout"], "Output_ADC"
    )
    if wo_dl: # without delay line
        stats.combine_per_component_energy(
            ["MRR"], "Photonic_Devices"
        )
    else:
        stats.combine_per_component_energy(
            ["MRR", "delay_line"], "Photonic_Devices"
        )
    stats.combine_per_component_energy(
        ["input_buffer", "output_buffer", "weight_buffer"], "SRAM"
    )
    # component energy
    df.loc[index, 'input_dac_energy'] = stats.per_component_energy["Input_DAC"]
    df.loc[index, 'photonic_devices_energy'] = stats.per_component_energy["Photonic_Devices"]
    df.loc[index, 'output_adc_energy'] = stats.per_component_energy["Output_ADC"]
    df.loc[index, 'sram_energy'] = stats.per_component_energy["SRAM"]
    # component power
    df.loc[index, 'input_dac_power'] = stats.per_component_energy["Input_DAC"] / stats.cycle_seconds / stats.cycles
    df.loc[index, 'photonic_devices_power'] = stats.per_component_energy["Photonic_Devices"] / stats.cycle_seconds / stats.cycles
    df.loc[index, 'output_adc_power'] = stats.per_component_energy["Output_ADC"] / stats.cycle_seconds / stats.cycles
    df.loc[index, 'sram_power'] = stats.per_component_energy["SRAM"] / stats.cycle_seconds / stats.cycles
    return df

def test_with_delay_line():
    df = pd.DataFrame()
    stats_with_delay_line = utils.quick_run("proposed", dnn='alexnet', layer='0', max_utilization=False, system="fetch_all_hbm2")
    stats_without_delay_line = utils.quick_run("proposed_no_dl", dnn='alexnet', layer='0', max_utilization=False, system="fetch_all_hbm2")

    # with delay line
    df = handle_results(stats_with_delay_line, df, "proposed_alexnet_0")
    df = handle_energy_area_power(stats_with_delay_line, df, "proposed_alexnet_0")
    df = handle_results_proposed(stats_with_delay_line, df, "proposed_alexnet_0")

    # without delay line
    df = handle_results(stats_without_delay_line, df, "proposed_no_dl_alexnet_0")
    df = handle_energy_area_power(stats_without_delay_line, df, "proposed_no_dl_alexnet_0")
    df = handle_results_proposed(stats_without_delay_line, df, "proposed_no_dl_alexnet_0", wo_dl=True)
    df.to_csv("results_proposed_with_delay_line_alexnet_0.csv")

def test_one_case():
    stats = utils.quick_run("proposed_albireo", dnn='alexnet', layer='0', max_utilization=False, system="fetch_all_hbm2")
    df = pd.DataFrame()
    df = handle_results(stats, df,  "albireo_alexnet_0")
    df = handle_energy_area_power(stats, df, "albireo_alexnet_0")
    df.to_csv("results_albireo_alexnet_0.csv")
    return df

def test_albireo(df: pd.DataFrame = None):

    if df is None:
        df = pd.DataFrame()
    
    stats = utils.quick_run("proposed", dnn='alexnet', layer='0', max_utilization=False, system="fetch_all_hbm2")
    df = pd.DataFrame()
    df = handle_results(stats, "proposed_alexnet_0", df)
    df = handle_results_proposed(stats, df, "proposed_alexnet_0")
    df.to_csv("results_proposed_alexnet_0.csv")
    stats = utils.parallel_test(
        utils.delayed(utils.quick_run)(
            "proposed_albireo", 
            variables={
                "SCALING": f'"{s}"',
            },
            dnn='mobilenet_v3', 
            layer='00', 
            max_utilization=False)
        for s in ["conservative", "moderate", "aggressive"]
    )
    for s, stat in zip(["conservative", "moderate", "aggressive"], stats):
        df = handle_results(stat, f"albireo_mobilenet_v3_00_{s}", df)
    
    df.to_csv("results_alberio.csv")
    return df

def daily_test():

    # run alberio
    ## alexnet
    stats = utils.quick_run("proposed", dnn='alexnet', layer='5', max_utilization=False, system="fetch_all_lpddr4")
    ## mobilenet_v3


def test_all_layers(df: pd.DataFrame = None):
    pass


def test_proposed(df: pd.DataFrame = None):

    if df is None:
        df = pd.DataFrame()
    
    stats = utils.parallel_test(
        utils.delayed(utils.quick_run)(
            "proposed", 
            variables={
                "SCALING": f'"{s}"',
                "N_COLUMNS": f'{n_rc}',
                "N_ROWS": f'{n_rc}',
                "N_TILES": f'{n_tile}',
            },
            dnn='mobilenet_v3', 
            layer='00', 
            max_utilization=False)
        for n_rc, n_tile in [(4, 8), (4, 16), (4, 32), (4, 64), (8, 4), (8, 8), (8, 16)]
        for s in ["conservative", "moderate", "aggressive"]
        # for s in ["aggressive"]
    )
    i = 0
    for n_rc, n_tile in [(4, 8), (4, 16), (4, 32), (4, 64), (8, 4), (8, 8), (8, 16)]:
        for s in zip(["conservative", "moderate", "aggressive"]):
        # for s in ["aggressive"]:
            stat = stats[i]
            i += 1
            df = handle_results(stat, f"proposed_mobilenet_v3_00_RC_{n_rc}_TILE_{n_tile}_{s}", df)
    df.to_csv("results_proposed.csv")
    return df

def test_ohas():
    stats = utils.quick_run("ohas_iccad_2021", dnn='mobilenet_v3', layer='00', max_utilization=False)
    print(stats.tops_per_w)

def test_sinangil():
    stats = utils.quick_run("sinangil_jssc_2021", dnn='mobilenet_v3', layer='00', max_utilization=False)
    print(stats.tops_per_w)

if __name__ == "__main__":
    daily_test()
    # test_albireo()
    
