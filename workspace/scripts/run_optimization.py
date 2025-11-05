# Todo: Optimize the architecture of ONN


import re
from matplotlib.pylab import f
from ocim_utils import fetch_all_layers, handle_all_results
import utils
from utils import bar_stacked, bar_side_by_side
import pandas as pd
import matplotlib.pyplot as plt

from arch_metrics_parser import load_metrics, analyze_relative_metrics, combine_metrics


def combine_stats_components(stats):
    stats.combine_per_component_area_energy(
        ["weight_mrr", "input_mrr"], "MRM",
    )
    stats.combine_per_component_area_energy(
        ["weight_dac", "input_dac"], "DAC",
    )
    stats.combine_per_component_area_energy(
        ["laser", "photodiode_output_readout", "TIA"], "OAC",
    )
    stats.combine_per_component_area_energy(
        ["adc"], "ADC",
    )
    stats.combine_per_component_area_energy(
        #["glb", "input_buffer", "output_buffer", "input_cache", "weight_cache"], "Cache",
        ["glb", "output_buffer"], "Cache",
    )
    stats.combine_per_component_area_energy(
        ["main_memory"], "Main Memory",
    )
    return stats

def optimize_mrr(dnn_name: str, system: str, save_name: str = "optimzation_mrr", onebit_input: bool = False, osa: bool = False):
    layer_paths = fetch_all_layers(dnn_name)
    macro_settings_configs = [
        (1, 1, 9, 113),
        (1, 1, 100, 12),
        (1, 64, 4, 4),
        (1, 32, 4, 8),
        (1, 16, 4, 16),
        (1, 8, 4, 32),
        (1, 32, 8, 4),
        (1, 16, 8, 8),
        (1, 8, 8, 16),
        (1, 4, 8, 32),
    ]
    if osa:
        macro = "proposed_mrr_optical_shift_add"
    else:
        macro = "proposed_mrr"
    stats = utils.parallel_test(
        utils.delayed(utils.quick_run)(
            macro=macro, 
            layer=l, 
            variables=dict(
                SCALING='"aggressive"',
                N_TILES=n_tiles,
                N_PES=n_pes,
                N_COLUMNS=n_columns,
                N_ROWS=n_rows,
                VOLTAGE_DAC_RESOLUTION=1 if onebit_input else 8,  # 1-bit input if specified
            ),
            system=system,
            max_utilization=False
        )
        for l in layer_paths
        for n_tiles, n_pes, n_columns, n_rows in macro_settings_configs
    ) 
    stats.clear_zero_energies()
    df = pd.DataFrame()

    output_postfix_str = ""
    output_postfix_str += "_1bit_input" if onebit_input else ""
    output_postfix_str += "_osa" if osa else ""

    # Save detailed breakdown results
    for idx_l, l in enumerate(layer_paths):
        for idx_cfg, (n_tiles, n_pes, n_columns, n_rows) in enumerate(macro_settings_configs):
            index = f"proposed_mrr_{dnn_name}_{l}_{n_tiles}tiles_{n_pes}pes_{n_columns}cols_{n_rows}rows"
            df = handle_all_results(stats[idx_l * len(macro_settings_configs) + idx_cfg], df, index)
    df.to_csv(f"./results/results_{save_name}_{dnn_name}_breakdown{output_postfix_str}.csv")

    # Plot energy breakdown layer by layer for each setting
    for idx_cfg, (n_tiles, n_pes, n_columns, n_rows) in enumerate(macro_settings_configs):
        fig, ax = plt.subplots(figsize=(10, 6))
        energy_breakdown = {
            l: stats[idx_l * len(macro_settings_configs) + idx_cfg].per_compute("per_component_energy")  / 1e-12
            for idx_l, l in enumerate(layer_paths)
        }
        bar_stacked(
            energy_breakdown,
            xlabel="Layer",
            ylabel="Energy (pJ/MAC)",
            title=f"Energy Breakdown for {n_tiles} Tiles, {n_pes} PEs, {n_columns} Columns, {n_rows} Rows",
            ax=ax,
        )
        # plt.tight_layout()
        plt.savefig(f"./results/energy_breakdown{output_postfix_str}_{save_name}_{dnn_name}_{n_tiles}tiles_{n_pes}pes_{n_columns}cols_{n_rows}rows.png", dpi=300)
        plt.savefig(f"./results/energy_breakdown{output_postfix_str}_{save_name}_{dnn_name}_{n_tiles}tiles_{n_pes}pes_{n_columns}cols_{n_rows}rows.pdf", dpi=300)
        plt.close(fig)

    # Plot latency breakdown layer by layer for each setting
    for idx_cfg, (n_tiles, n_pes, n_columns, n_rows) in enumerate(macro_settings_configs):
        fig, ax = plt.subplots(figsize=(10, 6))
        latency_breakdown = {
            l: stats[idx_l * len(macro_settings_configs) + idx_cfg].per_compute("latency")  / 1e-3
            for idx_l, l in enumerate(layer_paths)
        }
        bar_stacked(
            latency_breakdown,
            xlabel="Layer",
            ylabel="Latency (ms)",
            title=f"Latency Breakdown for {n_tiles} Tiles, {n_pes} PEs, {n_columns} Columns, {n_rows} Rows",
            ax=ax,
        )
        # plt.tight_layout()
        plt.savefig(f"./results/latency_breakdown{output_postfix_str}_{save_name}_{dnn_name}_{n_tiles}tiles_{n_pes}pes_{n_columns}cols_{n_rows}rows.png", dpi=300)
        plt.savefig(f"./results/latency_breakdown{output_postfix_str}_{save_name}_{dnn_name}_{n_tiles}tiles_{n_pes}pes_{n_columns}cols_{n_rows}rows.pdf", dpi=300)
        plt.close(fig)

    stats = combine_stats_components(stats)
    
    df = pd.DataFrame()
    for idx_l, l in enumerate(layer_paths):
        for idx_cfg, (n_tiles, n_pes, n_columns, n_rows) in enumerate(macro_settings_configs):
            index = f"proposed_mrr_{dnn_name}_{l}_{n_tiles}tiles_{n_pes}pes_{n_columns}cols_{n_rows}rows"
            df = handle_all_results(stats[idx_l * len(macro_settings_configs) + idx_cfg], df, index)

    df.to_csv(f"./results/results_{save_name}_{dnn_name}_combined{output_postfix_str}.csv")

    # plot overall latency for layers
 
    stats.aggregate_by("N_TILES", "N_PES", "N_COLUMNS", "N_ROWS")

    organized = {}
    for r in stats:
        key = []
        key.append(f'T{r.variables["N_TILES"]}')
        key.append(f'P{r.variables["N_PES"]}')
        key.append(f'C{r.variables["N_COLUMNS"]}')
        key.append(f'R{r.variables["N_ROWS"]}')
        key = ", ".join(key)
        organized[key] = r

    # Write metrics to CSV
    metrics_data = []
    for key, result in organized.items():
        metrics_data.append({
            'Architecture': key,
            'TOPS': result.tops,
            'Energy_per_MAC': result.energy,
            'Total_Area': result.area,
            'Total_Cycles': result.cycles,
            'Latency': result.latency,
            'Total_Power_W': result.energy / result.cycle_seconds / result.cycles,
            'TOPS_per_W': result.tops_per_w,
            'TOPS_per_mm2': result.tops_per_mm2,
            'EDP': result.energy * result.latency,
        })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(f"./results/architecture_metrics_{save_name}_{dnn_name}{output_postfix_str}.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(16, 9))

    bar_stacked(
        {k: v.per_compute("per_component_energy") / 1e-12 for k, v in organized.items()},
        xlabel="Architecture",
        ylabel="Energy (pJ/MAC)",
        title="Energy Efficiency",
        ax=axes[0],
    )

    bar_stacked(
        {k: v.per_component_energy / v.cycle_seconds / v.cycles for k, v in organized.items()},
        xlabel="Architecture",
        ylabel="Power (W)",
        title="Total Power",
        ax=axes[1],
    )

    # plot overall latency for layers
    bar_side_by_side(
        {k: {"": v.latency / 1e-3} for k, v in organized.items()},
        xlabel="Architecture",
        ylabel=f"Latency (ms)",
        title=f"Per-Architecture Latency",
        ax=axes[2],
    )

    # plot the EDP
    bar_side_by_side(
        {k: {"": v.energy * v.latency / 1e-9} for k, v in organized.items()},
        xlabel="Architecture",
        ylabel=f"EDP (pJ·ms)",
        title=f"Energy-Delay Product",
        ax=axes[3],
    )
    
    plt.tight_layout()
    plt.savefig(f"./results/{save_name}_{dnn_name}{output_postfix_str}.png", dpi=300)
    plt.savefig(f"./results/{save_name}_{dnn_name}{output_postfix_str}.pdf", dpi=300)

def optimize_mrr_1bit_input(dnn_name: str, system: str):
    layer_paths = fetch_all_layers(dnn_name)
    macro_settings_configs = [
        (1, 64, 4, 4),
        (2, 32, 4, 4),
        (4, 16, 4, 4),
        (8, 8, 4, 4),
        (16, 4, 4, 4),
        (32, 2, 4, 4),
        (64, 1, 4, 4),
        # (1, 32, 4, 8),
        # (2, 16, 4, 8),
        # (4, 8, 4, 8),
        # (8, 4, 4, 8),
        # (16, 2, 4, 8),
        # (32, 1, 4, 8), 
        # (1, 32, 8, 4),
        # (2, 16, 8, 4),
        # (4, 8, 8, 4),
        # (8, 4, 8, 4),
        # (16, 2, 8, 4),
        # (32, 1, 8, 4),
        (1, 16, 8, 8),
        (2, 8, 8, 8),
        (4, 4, 8, 8),
        (8, 2, 8, 8),
        (16, 1, 8, 8),
    ]
    scaling = "aggressive"  # Use aggressive scaling for 1-bit input
    stats = utils.parallel_test(
        utils.delayed(utils.quick_run)(
            macro="proposed_mrr", 
            layer=l, 
            variables=dict(
                N_TILES=n_tiles,
                N_PES=n_pes,
                N_COLUMNS=n_columns,
                N_ROWS=n_rows,
                VOLTAGE_DAC_RESOLUTION=1,  # 1-bit input
                SCALING='"aggressive"',  # Use aggressive scaling for 1-bit input
            ),
            system=system,
            max_utilization=False
        )
        for l in layer_paths
        for n_tiles, n_pes, n_columns, n_rows in macro_settings_configs
        
    ) 
    stats.combine_per_component_area_energy(
        ["weight_mrr", "input_mrr"], "Microring",
    )
    stats.combine_per_component_area_energy(
        ["weight_dac", "input_dac"], "EO",
    )
    stats.combine_per_component_area_energy(
        ["photodiode_output_readout", "TIA", "adc"], "OE",
    )
    stats.combine_per_component_area_energy(
        ["glb", "input_buffer", "output_buffer", "weight_cache"], "Memory",
    )
    stats.combine_per_component_area_energy(
        ["laser", "main_memory"], "Others",
    )
    stats.clear_zero_energies()
    df = pd.DataFrame()
    for idx_l, l in enumerate(layer_paths):
        for idx_cfg, (n_tiles, n_pes, n_columns, n_rows) in enumerate(macro_settings_configs):
            index = f"proposed_mrr_1bit_input_{dnn_name}_{l}_{n_tiles}tiles_{n_pes}pes_{n_columns}cols_{n_rows}rows"
            df = handle_all_results(stats[idx_l * len(macro_settings_configs) + idx_cfg], df, index)
    df.to_csv(f"./results/results_optimization_mrr_1bit_input_{dnn_name}_{scaling}.csv")

    stats.aggregate_by("N_TILES", "N_PES", "N_COLUMNS", "N_ROWS")

    organized = {}
    for r in stats:
        key = []
        key.append(f'T{r.variables["N_TILES"]}')
        key.append(f'P{r.variables["N_PES"]}')
        key.append(f'C{r.variables["N_COLUMNS"]}')
        key.append(f'R{r.variables["N_ROWS"]}')
        key = ", ".join(key)
        organized[key] = r

    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    bar_stacked(
        {k: v.per_compute("per_component_energy") for k, v in organized.items()},
        xlabel="Architecture",
        ylabel="Energy (pJ/MAC)",
        title="Per-Architecture Total Energy",
        ax=axes[0],
    )
    for ax, attrname, title, ylabel in [
        (axes[1], "tops_per_w", "Energy Efficiency", "TOPS/W"),
    ]:
        bar_side_by_side(
            {k: {"": getattr(v, attrname)} for k, v in organized.items()},
            xlabel="Architecture",
            ylabel=f"{title} ({ylabel})",
            title=f"Per-Architecture {title}",
            ax=ax,
        )
    plt.tight_layout()
    plt.savefig(f"./results/optimization_mrr_1bit_input_{dnn_name}_{scaling}.png", dpi=300)
    plt.savefig(f"./results/optimization_mrr_1bit_input_{dnn_name}_{scaling}.pdf", dpi=300)

def optimize_mrr_1bit_input_optical_shift_add(dnn_name: str, system: str):
    layer_paths = fetch_all_layers(dnn_name)
    macro_settings_configs = [
        # (1, 64, 4, 4),
        # (2, 32, 4, 4),
        # (4, 16, 4, 4),
        # (8, 8, 4, 4),
        # (16, 4, 4, 4),
        # (32, 2, 4, 4),
        # (64, 1, 4, 4),
        # (1, 32, 4, 8),
        # (2, 16, 4, 8),
        # (4, 8, 4, 8),
        # (8, 4, 4, 8),
        # (1, 32, 8, 4),
        # (2, 16, 8, 4),
        # (4, 8, 8, 4),
        # (8, 4, 8, 4),
        # (16, 2, 8, 4),
        # (32, 1, 8, 4),
        (1, 16, 8, 8),
        (2, 8, 8, 8),
        (4, 4, 8, 8),
        (8, 2, 8, 8),
        (16, 1, 8, 8),
    ]
    scaling = "aggressive"  # Use aggressive scaling for 1-bit input
    stats = utils.parallel_test(
        utils.delayed(utils.quick_run)(
            macro="proposed_mrr_optical_shift_add", 
            layer=l, 
            variables=dict(
                N_TILES=n_tiles,
                N_PES=n_pes,
                N_COLUMNS=n_columns,
                N_ROWS=n_rows,
                VOLTAGE_DAC_RESOLUTION=1,  # 1-bit input
                SCALING='"aggressive"',  # Use aggressive scaling for 1-bit input
            ),
            system=system,
            max_utilization=False
        )
        for l in layer_paths
        for n_tiles, n_pes, n_columns, n_rows in macro_settings_configs
        
    ) 
    stats.combine_per_component_area_energy(
        ["weight_mrr", "input_mrr"], "Microring",
    )
    stats.combine_per_component_area_energy(
        ["weight_dac", "input_dac"], "EO",
    )
    stats.combine_per_component_area_energy(
        ["photodiode_output_readout", "TIA", "adc"], "OE",
    )
    stats.combine_per_component_area_energy(
        ["glb", "input_buffer", "output_buffer", "weight_cache", "input_cache"], "Memory",
    )
    stats.combine_per_component_area_energy(
        ["laser", "main_memory"], "Others",
    )
    stats.clear_zero_energies()
    df = pd.DataFrame()
    for idx_l, l in enumerate(layer_paths):
        for idx_cfg, (n_tiles, n_pes, n_columns, n_rows) in enumerate(macro_settings_configs):
            index = f"proposed_mrr_1bit_input_osa_{dnn_name}_{l}_{n_tiles}tiles_{n_pes}pes_{n_columns}cols_{n_rows}rows"
            df = handle_all_results(stats[idx_l * len(macro_settings_configs) + idx_cfg], df, index)
    df.to_csv(f"./results/results_optimization_mrr_1bit_input_osa_{dnn_name}_{scaling}.csv")

    stats.aggregate_by("N_TILES", "N_PES", "N_COLUMNS", "N_ROWS")

    organized = {}
    for r in stats:
        key = []
        key.append(f'T{r.variables["N_TILES"]}')
        key.append(f'P{r.variables["N_PES"]}')
        key.append(f'C{r.variables["N_COLUMNS"]}')
        key.append(f'R{r.variables["N_ROWS"]}')
        key = ", ".join(key)
        organized[key] = r

    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    bar_stacked(
        {k: v.per_compute("per_component_energy") for k, v in organized.items()},
        xlabel="Architecture",
        ylabel="Energy (pJ/MAC)",
        title="Per-Architecture Total Energy",
        ax=axes[0],
    )
    for ax, attrname, title, ylabel in [
        (axes[1], "tops_per_w", "Energy Efficiency", "TOPS/W"),
    ]:
        bar_side_by_side(
            {k: {"": getattr(v, attrname)} for k, v in organized.items()},
            xlabel="Architecture",
            ylabel=f"{title} ({ylabel})",
            title=f"Per-Architecture {title}",
            ax=ax,
        )
    plt.tight_layout()
    plt.savefig(f"./results/optimization_mrr_1bit_input_osa_{dnn_name}_{scaling}.png", dpi=300)
    plt.savefig(f"./results/optimization_mrr_1bit_input_osa_{dnn_name}_{scaling}.pdf", dpi=300)

def report_arch_performance(macro_name: str, dnn_name: str, system: str, save_name: str, variables: dict = {}):
    layer_paths = fetch_all_layers(dnn_name)
    stats = utils.parallel_test(
        utils.delayed(utils.quick_run)(
            macro=macro_name,
            layer=l,
            variables=variables,
            system=system,
            max_utilization=False
        )
        for l in layer_paths
    )
    stats.aggregate_by("layer")

    stats.clear_zero_energies()

def analyze_architecture_results(dnn_name: str, onebit_input: bool = False, alpha: float = 1.0, beta: float = 1.5):
    """Analyze the architecture optimization results with multiple metrics."""
    def print_relative_analysis_results(dnn_name: str, relative_metrics: dict, combined_scores: dict):
        print(f"\nRelative Analysis Results for {dnn_name.upper()}:")
        print(f"{'Architecture':<20} {'Relative EDP':<15} {'Relative Latency':<20} {'Relative Energy/MAC':<25} {'Combined Score':<15}")
        print("-" * 100)
        for config in relative_metrics['edp'].keys():
            edp = relative_metrics['edp'][config]
            latency = relative_metrics['latency'][config]
            energy_per_mac = relative_metrics['energy_per_mac'][config]
            combined_score = combined_scores[config] 
            print(f"{config:<20} {edp:<15.4f} {latency:<20.4f} {energy_per_mac:<25.4f} {combined_score:<15.4f}")

        # export it into csv file
        df = pd.DataFrame({
            'Architecture': list(relative_metrics['edp'].keys()),
            'Relative EDP': list(relative_metrics['edp'].values()),
            'Relative Latency': list(relative_metrics['latency'].values()),
            'Relative Energy/MAC': list(relative_metrics['energy_per_mac'].values()),
            'Combined Score': list(combined_scores.values()),
        })
        df.to_csv(f"./results/relative_analysis_{dnn_name}{'_1bit_input' if onebit_input else ''}.csv", index=False)

    onebit_str = "_1bit_input" if onebit_input else ""
    metrics_file = f"./results/architecture_metrics_deapcnns_{dnn_name}{onebit_str}.csv"
    
    metrics = load_metrics(metrics_file)
    
    relative_metrics = analyze_relative_metrics(metrics)

    combined_scores = combine_metrics(relative_metrics, alpha=alpha, beta=beta)

    print_relative_analysis_results(dnn_name, relative_metrics, combined_scores)

    return combined_scores

def calculate_aggregated_score(scores_dict, lambda_param=0.25):
    """
    Calculate aggregated score across networks using geometric mean and worst-case bottleneck.
    Args:
        scores_dict: Dictionary of {network: score} pairs
        lambda_param: Weight for worst-case score (between 0 and 0.5)
    """

    
    # Equal weights for all networks
    weights = {net: 1.0 for net in scores_dict.keys()}
    weight_sum = sum(weights.values())
    
    # Calculate geometric mean component
    weighted_product = 1.0
    for net, score in scores_dict.items():
        weighted_product *= score ** weights[net]
    geometric_mean = weighted_product ** (1.0 / weight_sum)
    
    # Find worst-case score
    worst_score = max(scores_dict.values())
    
    # Combine using lambda parameter
    return geometric_mean * (1 - lambda_param) + worst_score * lambda_param

def optimize_all():
    # networks = ["alexnet", "vgg16", "resnet18", "mobilenet_v3", "gpt2_medium"]
    networks = ["alexnet"]
    for network in networks:
        optimize_mrr(network, "fetch_all_lpddr4", save_name="deapcnns", onebit_input=True, osa=True)
        optimize_mrr(network, "fetch_all_lpddr4", save_name="deapcnns", onebit_input=True, osa=False)

def analyze_all():
    def plot_combined_scores(networks, architecture_aggregated, combined_scores_all):
        # Print results in a well-formatted table
        print("\n" + "="*100)
        print("ARCHITECTURE OPTIMIZATION RESULTS")
        print("="*100)

        print(f"\n{'Architecture':<20} {'Combined Score':<15} {'Rank':<6}")
        print("-" * 45)

        # Sort architectures by aggregated score (descending)
        sorted_archs = sorted(architecture_aggregated.items(), key=lambda x: x[1], reverse=False)

        # Save aggregated scores to CSV
        aggregated_df = pd.DataFrame({
            'Architecture': [arch for arch, score in sorted_archs],
            'Aggregated_Score': [score for arch, score in sorted_archs],
            'Rank': list(range(1, len(sorted_archs) + 1))
        })
        aggregated_df.to_csv(f"./results/aggregated_architecture_scores_1bit_input.csv", index=False)

        # Save detailed scores by network to CSV
        detailed_data = []
        for network in networks:
            for arch, score in combined_scores_all[network].items():
                detailed_data.append({
                    'Network': network,
                    'Architecture': arch,
                    'Combined_Score': score
                })

        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_csv(f"./results/detailed_architecture_scores_by_network_1bit_input.csv", index=False)

        for rank, (arch, score) in enumerate(sorted_archs, 1):
            print(f"{arch:<20} {score:<15.4f} {rank:<6}")

        print("\n" + "="*100)
        print("DETAILED SCORES BY NETWORK")
        print("="*100)

        # Print detailed scores for each network
        for network in networks:
            print(f"\n{network.upper()}: ")
            print(f"{'Architecture':<20} {'Score':<10}")
            print("-" * 32)
            
            network_scores = combined_scores_all[network]
            sorted_network = sorted(network_scores.items(), key=lambda x: x[1], reverse=False)
            
            for arch, score in sorted_network:
                print(f"{arch:<20} {score:<10.4f}")

        print("\n" + "="*50)
        print("BEST ARCHITECTURE OVERALL:")
        print(f"{sorted_archs[0][0]} (Score: {sorted_archs[0][1]:.4f})")
        print("="*50)


    """Analyze architecture results across networks using geometric mean and worst-case analysis."""
    networks = ["alexnet", "vgg16", "resnet18", "mobilenet_v3", "gpt2_medium"]
    combined_scores_all = {}
    
    # Get scores for each network
    for network in networks:
        combined_scores_all[network] = analyze_architecture_results(network, onebit_input=True)

    # Calculate aggregated scores for each architecture
    architecture_aggregated = {}
    all_architectures = set()
    for scores in combined_scores_all.values():
        all_architectures.update(scores.keys())

    for arch in all_architectures:
        arch_scores = {net: scores.get(arch, 0) for net, scores in combined_scores_all.items() if arch in scores}
        if arch_scores:
            architecture_aggregated[arch] = calculate_aggregated_score(arch_scores)

    plot_combined_scores(networks, architecture_aggregated, combined_scores_all)
    



if __name__ == "__main__":
    # Test with different networks
    optimize_all()
    # analyze_all()

