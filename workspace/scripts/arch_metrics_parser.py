import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ArchMetrics:
    tiles: int
    pes: int
    columns: int
    rows: int
    tops: float
    energy_per_mac: float
    total_area: float
    total_cycles: int
    latency: float
    total_power: float
    tops_per_w: float
    tops_per_mm2: float
    edp: float

    @property
    def config_str(self) -> str:
        return f"T{self.tiles}, P{self.pes}, C{self.columns}, R{self.rows}"
    
    @staticmethod
    def get_baseline_config(pes: int) -> str:
        return f"T1, P{pes}, C4, R4"

def parse_architecture_string(arch_str: str) -> Tuple[int, int, int, int]:
    """Parse architecture string like 'T1, P32, C4, R8' into component values."""
    parts = arch_str.replace('"', '').split(', ')
    return (
        int(parts[0][1:]),  # tiles
        int(parts[1][1:]),  # pes
        int(parts[2][1:]),  # columns
        int(parts[3][1:])   # rows
    )

def load_metrics(filepath: str) -> Dict[str, ArchMetrics]:
    """Load architecture metrics from CSV file into a dictionary."""
    df = pd.read_csv(filepath)
    metrics = {}
    
    for _, row in df.iterrows():
        tiles, pes, columns, rows = parse_architecture_string(row['Architecture'])
        
        metric = ArchMetrics(
            tiles=tiles,
            pes=pes,
            columns=columns,
            rows=rows,
            tops=row['TOPS'],
            energy_per_mac=row['Energy_per_MAC'],
            total_area=row['Total_Area'],
            total_cycles=row['Total_Cycles'],
            latency=row['Latency'],
            total_power=row['Total_Power_W'],
            tops_per_w=row['TOPS_per_W'],
            tops_per_mm2=row['TOPS_per_mm2'],
            edp=row['EDP']
        )
        
        metrics[metric.config_str] = metric
    
    return metrics

def analyze_relative_metrics(metrics: Dict[str, ArchMetrics]) -> Dict[str, Dict[str, float]]:
    """Analyze architectures by multiple criteria and return sorted results.
    
    Args:
        metrics: Dictionary of architecture metrics
    
    Returns:
        Dictionary with criteria as keys and list of (architecture, value) tuples sorted by value
    """
    results = {}
    relative_metrics = {"edp": {}, "latency": {}, "energy_per_mac": {}}

    # Identify baseline configuration (4x4)
    for config, metric in metrics.items():
        if metric.columns == 4 and metric.rows == 4:
            baseline_config = config

    baseline_edp = metrics[baseline_config].edp
    baseline_latency = metrics[baseline_config].latency
    baseline_energy_per_mac = metrics[baseline_config].energy_per_mac
    
    for config, metric in metrics.items():
        relative_metrics["edp"][config] = metric.edp / baseline_edp
        relative_metrics["latency"][config] = metric.latency / baseline_latency
        relative_metrics["energy_per_mac"][config] = metric.energy_per_mac / baseline_energy_per_mac
    
    return relative_metrics
    
def combine_metrics(relative_metrics: Dict[str, Dict[str, float]], alpha: float = 1.0, beta: float = 1.5) -> Dict[str, float]:
    lat_metrics = relative_metrics["latency"]
    epm_metrics = relative_metrics["energy_per_mac"]
    combined_scores = {}

    # Combined scoring: lower is better (latency^alpha * edp^beta)
    for config in lat_metrics.keys():
        combined_score = (lat_metrics[config] ** alpha) * (epm_metrics[config] ** beta)
        combined_scores[config] = combined_score

    return combined_scores

def print_analysis_results(dnn_name: str, analysis_results: Dict[str, List[Tuple[str, float]]]):
    """Print analysis results in a formatted way."""
    print(f"\nArchitecture Analysis Results for {dnn_name}")
    print("=" * 60)
    
    metric_labels = {
        'tops': 'Performance (TOPS)',
        'tops_per_w': 'Energy Efficiency (TOPS/W)', 
        'tops_per_mm2': 'Area Efficiency (TOPS/mm²)',
        'edp': 'Energy-Delay Product (pJ·s)',
        'latency': 'Latency (s)',
        'total_power': 'Power Consumption (W)'
    }
    
    for metric, results in analysis_results.items():
        print(f"\n{metric_labels.get(metric, metric)}:")
        print("-" * 40)
        for arch, value in results[:5]:  # Show top 5 for each metric
            print(f"{arch}: {value:.3f}")
