from matplotlib.pylab import f
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_and_clean_data(csv_path):
    """Load CSV data and clean column names and workload identifiers"""
    df = pd.read_csv(csv_path, index_col=0)
    # Extract workload numbers from index for cleaner labels
    df['workload'] = df.index.str.split('/').str[-1].str.replace('.yaml', '').astype(int)
    df = df.sort_values('workload')
    return df

def load_multiple_architectures(arch_configs):
    """Load and clean data for multiple architectures"""
    arch_data = {}
    for arch_name, csv_path in arch_configs.items():
        df = load_and_clean_data(csv_path)
        arch_data[arch_name] = df
    return arch_data

def plot_architecture_performance_comparison(arch_data, output_dir):
    """Compare overall performance metrics across architectures using bar charts"""
    metrics = ['tops', 'energy', 'latency', 'tops_per_w']
    workloads = None

    # Get common workloads across all architectures
    for arch_name, df in arch_data.items():
        if workloads is None:
            workloads = set(df['workload'])
        else:
            workloads = workloads.intersection(set(df['workload']))

    workloads = sorted(list(workloads))
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        bar_width = 0.2  # Width of each bar
        x_positions = np.arange(len(workloads))  # X positions for workloads
        
        for j, (arch_name, df) in enumerate(arch_data.items()):
            workload_data = df[df['workload'].isin(workloads)].sort_values('workload')
            if metric == 'latency':
                values = workload_data[metric] * 1e6  # Convert to microseconds
                ylabel = 'Latency (μs)'
            else:
                values = workload_data[metric]
                ylabel = metric.replace('_', '/').upper() if '_' in metric else metric.upper()
            
            # Adjust bar positions for each architecture
            bar_positions = x_positions + j * bar_width
            axes[i].bar(bar_positions, values, width=bar_width, label=arch_name)
        
        axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[i].set_xlabel('Workload')
        axes[i].set_ylabel(ylabel)
        axes[i].set_xticks(x_positions + bar_width * (len(arch_data) - 1) / 2)
        axes[i].set_xticklabels(workloads)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'architecture_performance_comparison.png'))
    plt.close()

    # plot the performace of combined workload in 2x2 layout with value annotations
    combined_metrics = {}
    baseline_arch = list(arch_data.keys())[0]  # Use first architecture as baseline
    
    for arch_name, df in arch_data.items():
        combined_metrics[arch_name] = {
            'tops': df['tops'].sum(),
            'energy': df['energy'].sum(),
            'latency': df['latency'].sum() * 1e6,  # Convert to microseconds
            'tops_per_w': df['tops'].sum() / (df['energy'] / df['latency']).sum()
        }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [combined_metrics[arch_name][metric] for arch_name in arch_data.keys()]
        baseline_value = combined_metrics[baseline_arch][metric]
        
        if metric == 'latency':
            ylabel = 'Latency (μs)'
        else:
            ylabel = metric.replace('_', '/').upper() if '_' in metric else metric.upper()
        
        bars = axes[i].bar(arch_data.keys(), values, color=sns.color_palette("husl", len(arch_data)))
        axes[i].set_title(f'Combined {metric.replace("_", " ").title()}')
        axes[i].set_ylabel(ylabel)
        axes[i].grid(True, alpha=0.3)
        
        # Add value annotations and percentage change
        for j, (bar, arch_name) in enumerate(zip(bars, arch_data.keys())):
            height = bar.get_height()
            
            # Calculate percentage change from baseline
            if arch_name == baseline_arch:
                change_text = '(baseline)'
            else:
                if metric == 'latency' or metric == 'energy':  # Lower is better
                    change = ((baseline_value - height) / baseline_value) * 100
                    change_text = f'({change:+.1f}%)'
                else:  # Higher is better (tops, tops_per_w)
                    change = ((height - baseline_value) / baseline_value) * 100
                    change_text = f'({change:+.1f}%)'
            
            # Annotate with value and change
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}\n{change_text}',
                        ha='center', va='bottom', fontsize=9)
        
        # Rotate x-axis labels if needed
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_architecture_performance.png'))
    plt.close()

def compare_workload(workload_id):
    arch_configs = {
        'HolyLight': f'/home/zhanghf/projects/cimloop/results/{workload_id}/results_holylight_date_2019.csv',
        'proposed': f'/home/zhanghf/projects/cimloop/results/{workload_id}/results_proposed_mrr.csv',
        '1BitInput': f'/home/zhanghf/projects/cimloop/results/{workload_id}/results_proposed_mrr_1bit_input.csv',
        '1BitInputCache': f'/home/zhanghf/projects/cimloop/results/{workload_id}/results_proposed_mrr_1bit_input_delay_line.csv',
        'Hybrid': f'/home/zhanghf/projects/cimloop/results/{workload_id}/results_proposed_mrr_1bit_input_hybrid.csv',

    }
    output_dir = f'/home/zhanghf/projects/cimloop/results/{workload_id}/plots'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load multiple architectures
    arch_data = load_multiple_architectures(arch_configs)
        
    # Compare performance across architectures
    plot_architecture_performance_comparison(arch_data, output_dir)
            
def main():
    # workloads = ['alexnet', 'mobilenet_v3', 'resnet18_condensed', 'vgg16_condensed']
    workloads = ['alexnet', 'resnet18', ]
    for workload in workloads:  
        compare_workload(workload)    
if __name__ == "__main__":
    main()
