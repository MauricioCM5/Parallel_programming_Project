import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

class GraphColoringAnalyzer:
    def __init__(self):
        self.dataset_info = {
            'Queen4147': {
                'type': 'PDE Problem',
                'vertices': 4.1e6,
                'edges': 163e6,
                'avg_degree': 78,
                'max_degree': 89,
                'performance': {
                    'D1': {'execution_time': 0.5, 'color_count': 89},
                    'D1-2GL': {'execution_time': 0.42, 'color_count': 91},
                    'D2': {'execution_time': 0.6, 'color_count': 178},
                    'PD2': {'execution_time': 0.55, 'color_count': 156}
                }
            },
            'com-Friendster': {
                'type': 'Social Network',
                'vertices': 66e6,
                'edges': 1.8e9,
                'avg_degree': 55,
                'max_degree': 5200,
                'performance': {
                    'D1': {'execution_time': 1.2, 'color_count': 123},
                    'D1-2GL': {'execution_time': 1.0, 'color_count': 125},
                    'D2': {'execution_time': 1.8, 'color_count': 246},
                    'PD2': {'execution_time': 1.5, 'color_count': 198}
                }
            },
            'twitter7': {
                'type': 'Social Network',
                'vertices': 42e6,
                'edges': 1.4e9,
                'avg_degree': 35,
                'max_degree': 2.9e6,
                'performance': {
                    'D1': {'execution_time': 0.9, 'color_count': 145},
                    'D1-2GL': {'execution_time': 0.85, 'color_count': 148},
                    'D2': {'execution_time': 1.4, 'color_count': 290},
                    'PD2': {'execution_time': 1.2, 'color_count': 235}
                }
            },
            'europe_osm': {
                'type': 'Road Network',
                'vertices': 51e6,
                'edges': 54e6,
                'avg_degree': 2.1,
                'max_degree': 13,
                'performance': {
                    'D1': {'execution_time': 0.3, 'color_count': 4},
                    'D1-2GL': {'execution_time': 0.28, 'color_count': 4},
                    'D2': {'execution_time': 0.45, 'color_count': 8},
                    'PD2': {'execution_time': 0.4, 'color_count': 6}
                }
            }
        }

    def analyze_graph(self, graph_name: str) -> Dict[str, List[Tuple[float, int]]]:
        algorithms = list(self.dataset_info[graph_name]['performance'].keys())
        results = {algo: [] for algo in algorithms}
        
        print(f"\nAnalyzing performance data for {graph_name}")
        print(f"Graph type: {self.dataset_info[graph_name]['type']}")
        print(f"Vertices: {self.dataset_info[graph_name]['vertices']:,.0f}")
        print(f"Edges: {self.dataset_info[graph_name]['edges']:,.0f}")
        
        for algo in algorithms:
            perf_data = self.dataset_info[graph_name]['performance'][algo]
            results[algo].append((perf_data['execution_time'], perf_data['color_count']))
            print(f"\nAlgorithm: {algo}")
            print(f"  Execution time: {perf_data['execution_time']:.2f}s")
            print(f"  Colors used: {perf_data['color_count']}")
                
        return results
    
    def plot_performance(self, results: Dict[str, List[Tuple[float, int]]], graph_name: str):
        algorithms = list(results.keys())
        
        execution_times = [r[0][0] for r in results.values()]
        colors_used = [r[0][1] for r in results.values()]
        
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(algorithms))
        width = 0.35
        ax1.bar(x, execution_times, width)
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Times')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(x, colors_used, width)
        ax2.set_ylabel('Number of Colors')
        ax2.set_title('Color Usage')
        ax2.set_xticks(x)
        ax2.set_xticklabels(algorithms)
        
        ax3 = fig.add_subplot(gs[1, 0])
        baseline_time = min(execution_times)
        time_ratios = [t/baseline_time for t in execution_times]
        ax3.bar(algorithms, time_ratios)
        ax3.set_ylabel('Relative Performance')
        ax3.set_title('Performance Comparison')
        for i, ratio in enumerate(time_ratios):
            ax3.text(i, ratio, f'{ratio:.1f}x', ha='center', va='bottom')
        
        ax4 = fig.add_subplot(gs[1, 1])
        optimal_colors = min(colors_used)
        color_ratios = [c/optimal_colors for c in colors_used]
        ax4.bar(algorithms, color_ratios)
        ax4.set_ylabel('Color Efficiency')
        ax4.set_title('Coloring Efficiency')
        ax4.axhline(y=1, color='r', linestyle='--', alpha=0.3)
        for i, ratio in enumerate(color_ratios):
            ax4.text(i, ratio, f'{ratio:.2f}x', ha='center', va='bottom')
        
        plt.suptitle(f'Performance Analysis - {graph_name}\n' +
                    f'({self.dataset_info[graph_name]["type"]}, ' +
                    f'{self.dataset_info[graph_name]["vertices"]:,.0f} vertices)',
                    y=1.02)
        plt.tight_layout()
        plt.show()
        
    def plot_graph_comparison(self):
        graphs = list(self.dataset_info.keys())
        algorithms = ['D1', 'D1-2GL', 'D2', 'PD2']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        x = np.arange(len(graphs))
        width = 0.2
        
        for i, algo in enumerate(algorithms):
            times = [self.dataset_info[g]['performance'][algo]['execution_time'] for g in graphs]
            ax1.bar(x + i*width - width*1.5, times, width, label=algo)
        
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Execution Time by Graph Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels(graphs, rotation=45)
        ax1.legend()
        
        for i, algo in enumerate(algorithms):
            colors = [self.dataset_info[g]['performance'][algo]['color_count'] for g in graphs]
            ax2.bar(x + i*width - width*1.5, colors, width, label=algo)
        
        ax2.set_ylabel('Number of Colors')
        ax2.set_title('Color Usage by Graph Type')
        ax2.set_xticks(x)
        ax2.set_xticklabels(graphs, rotation=45)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def print_analysis(self, results: Dict[str, List[Tuple[float, int]]], graph_name: str):
        algorithms = list(results.keys())
        print(f"\nDetailed Analysis for {graph_name}:")
        print(f"Graph type: {self.dataset_info[graph_name]['type']}")
        print(f"Vertices: {self.dataset_info[graph_name]['vertices']:,.0f}")
        print(f"Edges: {self.dataset_info[graph_name]['edges']:,.0f}")
        print(f"Average degree: {self.dataset_info[graph_name]['avg_degree']}")
        print(f"Maximum degree: {self.dataset_info[graph_name]['max_degree']}")
        
        print("\nAlgorithm Performance:")
        optimal_time = min(r[0][0] for r in results.values())
        optimal_colors = min(r[0][1] for r in results.values())
        
        for algo in algorithms:
            time = results[algo][0][0]
            colors = results[algo][0][1]
            
            print(f"\n{algo}:")
            print(f"  Execution time: {time:.2f}s")
            print(f"  Performance ratio: {time/optimal_time:.2f}x optimal")
            print(f"  Colors used: {colors}")
            print(f"  Color efficiency: {colors/optimal_colors:.2f}x optimal")

analyzer = GraphColoringAnalyzer()
results = analyzer.analyze_graph("Queen4147")
analyzer.plot_performance(results, "Queen4147")
analyzer.print_analysis(results, "Queen4147")
analyzer.plot_graph_comparison()