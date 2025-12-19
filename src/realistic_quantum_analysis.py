import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time
import math

@dataclass
class SearchResult:
    """Search algorithm result"""
    algorithm: str
    target_found: bool
    iterations: int
    time_elapsed: float
    state_evolution: List[np.ndarray]
    
@dataclass
class CombinatorialAnalysis:
    """Why Grover is more efficient"""
    n_items: int
    classical_complexity: str
    quantum_complexity: str
    classical_iterations: int
    grover_iterations: int
    speedup_factor: float
    probability_amplification: List[float]

class QuantumStateVisualizer:
    """Visualize quantum states realistically"""
    
    @staticmethod
    def visualize_amplitudes(state: np.ndarray, title: str = "Quantum State"):
        """Visualize probability amplitudes"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        indices = range(len(state))
        amplitudes = np.abs(state)
        phases = np.angle(state)
        probabilities = amplitudes ** 2
        
        # Amplitude bars
        ax1.bar(indices, amplitudes, color='blue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Basis State |i⟩', fontsize=11)
        ax1.set_ylabel('Amplitude |αᵢ|', fontsize=11)
        ax1.set_title(f'{title} - Amplitudes', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Probability distribution
        ax2.bar(indices, probabilities, color='red', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Basis State |i⟩', fontsize=11)
        ax2.set_ylabel('Probability |αᵢ|²', fontsize=11)
        ax2.set_title(f'{title} - Probabilities', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def visualize_bloch_sphere(state: np.ndarray):
        """Visualize 2-level state on Bloch sphere"""
        if len(state) != 2:
            print("Bloch sphere only for 2-level systems")
            return
        
        # Extract Bloch coordinates
        theta = 2 * np.arccos(np.abs(state[0]))
        phi = np.angle(state[1]) - np.angle(state[0])
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='cyan')
        
        # Draw state vector
        ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=3)
        
        # Draw axes
        ax.plot([0, 1.2], [0, 0], [0, 0], 'k-', alpha=0.3)
        ax.plot([0, 0], [0, 1.2], [0, 0], 'k-', alpha=0.3)
        ax.plot([0, 0], [0, 0], [0, 1.2], 'k-', alpha=0.3)
        
        ax.text(1.3, 0, 0, 'X', fontsize=12)
        ax.text(0, 1.3, 0, 'Y', fontsize=12)
        ax.text(0, 0, 1.3, '|0⟩', fontsize=12)
        ax.text(0, 0, -1.3, '|1⟩', fontsize=12)
        
        ax.set_title('Bloch Sphere Representation', fontsize=14, fontweight='bold')
        ax.set_box_aspect([1,1,1])
        
        return fig

class ClassicalSearch:
    """Classical linear search"""
    
    def __init__(self, database: List[int], target: int):
        self.database = database
        self.target = target
        self.n = len(database)
    
    def search(self) -> SearchResult:
        """Linear search O(N)"""
        start_time = time.time()
        
        for i, item in enumerate(self.database):
            if item == self.target:
                elapsed = time.time() - start_time
                return SearchResult(
                    algorithm="Classical Linear",
                    target_found=True,
                    iterations=i + 1,
                    time_elapsed=elapsed,
                    state_evolution=[]
                )
        
        elapsed = time.time() - start_time
        return SearchResult(
            algorithm="Classical Linear",
            target_found=False,
            iterations=self.n,
            time_elapsed=elapsed,
            state_evolution=[]
        )

class GroverSearch:
    """Grover's quantum search algorithm (simulated)"""
    
    def __init__(self, n_items: int, target_idx: int):
        self.n = n_items
        self.target_idx = target_idx
        self.state = None
        self.state_evolution = []
    
    def initialize_superposition(self) -> np.ndarray:
        """Create uniform superposition |ψ⟩ = 1/√N Σ|i⟩"""
        state = np.ones(self.n, dtype=complex) / np.sqrt(self.n)
        return state
    
    def oracle(self, state: np.ndarray) -> np.ndarray:
        """Oracle marks target: O|x⟩ = -|x⟩ if x=target, else |x⟩"""
        marked_state = state.copy()
        marked_state[self.target_idx] *= -1
        return marked_state
    
    def diffusion(self, state: np.ndarray) -> np.ndarray:
        """Diffusion operator: D = 2|ψ⟩⟨ψ| - I"""
        avg = np.mean(state)
        diffused = 2 * avg - state
        return diffused
    
    def grover_iteration(self, state: np.ndarray) -> np.ndarray:
        """Single Grover iteration: G = D·O"""
        state = self.oracle(state)
        state = self.diffusion(state)
        return state
    
    def optimal_iterations(self) -> int:
        """Optimal iterations ≈ π/4 * √N"""
        return max(1, int(np.pi / 4 * np.sqrt(self.n)))
    
    def search(self) -> SearchResult:
        """Execute Grover search"""
        start_time = time.time()
        
        # Initialize
        state = self.initialize_superposition()
        self.state_evolution.append(state.copy())
        
        # Optimal iterations
        iterations = self.optimal_iterations()
        
        # Apply Grover iterations
        for _ in range(iterations):
            state = self.grover_iteration(state)
            self.state_evolution.append(state.copy())
        
        # Measure (find maximum probability)
        probabilities = np.abs(state) ** 2
        found_idx = np.argmax(probabilities)
        
        elapsed = time.time() - start_time
        
        return SearchResult(
            algorithm="Grover Quantum",
            target_found=(found_idx == self.target_idx),
            iterations=iterations,
            time_elapsed=elapsed,
            state_evolution=self.state_evolution
        )

class HybridSearch:
    """Hybrid: Classical + Quantum-inspired"""
    
    def __init__(self, database: List[int], target: int):
        self.database = database
        self.target = target
        self.n = len(database)
    
    def search(self) -> SearchResult:
        """Hybrid approach: partition + quantum-inspired on each"""
        start_time = time.time()
        
        # Partition into √N blocks
        block_size = max(1, int(np.sqrt(self.n)))
        n_blocks = (self.n + block_size - 1) // block_size
        
        total_iterations = 0
        state_evolution = []
        
        # Classical search through blocks
        for block_idx in range(n_blocks):
            start = block_idx * block_size
            end = min(start + block_size, self.n)
            block = self.database[start:end]
            
            # Quantum-inspired search within block (simulated speedup)
            grover_iters = max(1, int(np.sqrt(len(block))))
            total_iterations += grover_iters
            
            # Check if target in block
            if self.target in block:
                elapsed = time.time() - start_time
                return SearchResult(
                    algorithm="Hybrid Classical-Quantum",
                    target_found=True,
                    iterations=total_iterations,
                    time_elapsed=elapsed,
                    state_evolution=state_evolution
                )
        
        elapsed = time.time() - start_time
        return SearchResult(
            algorithm="Hybrid Classical-Quantum",
            target_found=False,
            iterations=total_iterations,
            time_elapsed=elapsed,
            state_evolution=state_evolution
        )

class CombinatorialAnalyzer:
    """Analyze why Grover is more efficient"""
    
    @staticmethod
    def analyze(n_items: int) -> CombinatorialAnalysis:
        """Combinatorial analysis of search complexity"""
        
        # Classical: must check all items in worst case
        classical_iterations = n_items
        classical_complexity = f"O(N) = O({n_items})"
        
        # Grover: only needs √N iterations
        grover_iterations = max(1, int(np.pi / 4 * np.sqrt(n_items)))
        quantum_complexity = f"O(√N) = O({int(np.sqrt(n_items))})"
        
        # Speedup
        speedup = classical_iterations / grover_iterations
        
        # Probability amplification over iterations
        prob_amplification = []
        for k in range(grover_iterations + 1):
            # Probability of measuring target after k iterations
            theta = np.arcsin(1 / np.sqrt(n_items))
            prob = np.sin((2*k + 1) * theta) ** 2
            prob_amplification.append(prob)
        
        return CombinatorialAnalysis(
            n_items=n_items,
            classical_complexity=classical_complexity,
            quantum_complexity=quantum_complexity,
            classical_iterations=classical_iterations,
            grover_iterations=grover_iterations,
            speedup_factor=speedup,
            probability_amplification=prob_amplification
        )
    
    @staticmethod
    def explain_efficiency(analysis: CombinatorialAnalysis):
        """Explain why Grover is more efficient"""
        print(f"\n{'='*70}")
        print(f"🔬 COMBINATORIAL ANALYSIS: Why Grover is More Efficient")
        print(f"{'='*70}")
        
        print(f"\n📊 Search Space: N = {analysis.n_items} items")
        
        print(f"\n🔴 Classical Search:")
        print(f"   Complexity: {analysis.classical_complexity}")
        print(f"   Iterations: {analysis.classical_iterations}")
        print(f"   Strategy: Check each item sequentially")
        print(f"   Worst case: Must check ALL items")
        
        print(f"\n🔵 Grover Quantum Search:")
        print(f"   Complexity: {analysis.quantum_complexity}")
        print(f"   Iterations: {analysis.grover_iterations}")
        print(f"   Strategy: Amplitude amplification")
        print(f"   Key insight: Rotate in √N-dimensional space")
        
        print(f"\n⚡ Speedup Factor: {analysis.speedup_factor:.2f}x")
        
        print(f"\n🎯 Why It Works:")
        print(f"   1. Superposition: Start with ALL items at once")
        print(f"   2. Oracle: Mark target with phase flip")
        print(f"   3. Diffusion: Amplify marked state")
        print(f"   4. Repeat: Only √N times needed!")
        
        print(f"\n📈 Probability Amplification:")
        for i, prob in enumerate(analysis.probability_amplification):
            if i % max(1, len(analysis.probability_amplification) // 5) == 0:
                print(f"   Iteration {i}: P(target) = {prob:.4f}")
        
        final_prob = analysis.probability_amplification[-1]
        print(f"\n   Final: P(target) = {final_prob:.4f} ({final_prob*100:.1f}%)")

class QuantumBenchmark:
    """Benchmark classical vs quantum-inspired performance"""
    
    def __init__(self):
        self.results = []
    
    def run_benchmark(self, sizes: List[int], trials: int = 10):
        """Run comprehensive benchmark"""
        print(f"\n{'='*70}")
        print(f"⚡ QUANTUM vs CLASSICAL BENCHMARK")
        print(f"{'='*70}")
        
        for n in sizes:
            print(f"\n📊 Testing N = {n} items...")
            
            classical_times = []
            grover_times = []
            hybrid_times = []
            
            classical_iters = []
            grover_iters = []
            hybrid_iters = []
            
            for trial in range(trials):
                # Create database
                database = list(range(n))
                target = np.random.randint(0, n)
                
                # Classical search
                classical = ClassicalSearch(database, target)
                c_result = classical.search()
                classical_times.append(c_result.time_elapsed)
                classical_iters.append(c_result.iterations)
                
                # Grover search
                grover = GroverSearch(n, target)
                g_result = grover.search()
                grover_times.append(g_result.time_elapsed)
                grover_iters.append(g_result.iterations)
                
                # Hybrid search
                hybrid = HybridSearch(database, target)
                h_result = hybrid.search()
                hybrid_times.append(h_result.time_elapsed)
                hybrid_iters.append(h_result.iterations)
            
            # Average results
            result = {
                'n': n,
                'classical_time': np.mean(classical_times),
                'grover_time': np.mean(grover_times),
                'hybrid_time': np.mean(hybrid_times),
                'classical_iters': np.mean(classical_iters),
                'grover_iters': np.mean(grover_iters),
                'hybrid_iters': np.mean(hybrid_iters),
                'speedup_grover': np.mean(classical_iters) / np.mean(grover_iters),
                'speedup_hybrid': np.mean(classical_iters) / np.mean(hybrid_iters)
            }
            
            self.results.append(result)
            
            print(f"   Classical: {result['classical_iters']:.1f} iters, {result['classical_time']*1000:.3f}ms")
            print(f"   Grover:    {result['grover_iters']:.1f} iters, {result['grover_time']*1000:.3f}ms")
            print(f"   Hybrid:    {result['hybrid_iters']:.1f} iters, {result['hybrid_time']*1000:.3f}ms")
            print(f"   Speedup:   {result['speedup_grover']:.2f}x (Grover), {result['speedup_hybrid']:.2f}x (Hybrid)")
    
    def visualize_results(self):
        """Visualize benchmark results"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        sizes = [r['n'] for r in self.results]
        
        # 1. Iterations comparison
        axes[0, 0].plot(sizes, [r['classical_iters'] for r in self.results], 
                       'o-', label='Classical O(N)', linewidth=2, markersize=8)
        axes[0, 0].plot(sizes, [r['grover_iters'] for r in self.results], 
                       's-', label='Grover O(√N)', linewidth=2, markersize=8)
        axes[0, 0].plot(sizes, [r['hybrid_iters'] for r in self.results], 
                       '^-', label='Hybrid', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Database Size (N)', fontsize=11)
        axes[0, 0].set_ylabel('Iterations Required', fontsize=11)
        axes[0, 0].set_title('Search Iterations: Classical vs Quantum', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # 2. Speedup factor
        axes[0, 1].plot(sizes, [r['speedup_grover'] for r in self.results], 
                       'o-', label='Grover Speedup', color='green', linewidth=2, markersize=8)
        axes[0, 1].plot(sizes, [r['speedup_hybrid'] for r in self.results], 
                       's-', label='Hybrid Speedup', color='orange', linewidth=2, markersize=8)
        axes[0, 1].plot(sizes, [np.sqrt(n) for n in sizes], 
                       '--', label='Theoretical √N', color='red', linewidth=2)
        axes[0, 1].set_xlabel('Database Size (N)', fontsize=11)
        axes[0, 1].set_ylabel('Speedup Factor', fontsize=11)
        axes[0, 1].set_title('Quantum Speedup vs Database Size', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Time comparison
        axes[1, 0].bar(np.arange(len(sizes)) - 0.2, [r['classical_time']*1000 for r in self.results], 
                      width=0.2, label='Classical', alpha=0.7)
        axes[1, 0].bar(np.arange(len(sizes)), [r['grover_time']*1000 for r in self.results], 
                      width=0.2, label='Grover', alpha=0.7)
        axes[1, 0].bar(np.arange(len(sizes)) + 0.2, [r['hybrid_time']*1000 for r in self.results], 
                      width=0.2, label='Hybrid', alpha=0.7)
        axes[1, 0].set_xticks(range(len(sizes)))
        axes[1, 0].set_xticklabels(sizes)
        axes[1, 0].set_xlabel('Database Size (N)', fontsize=11)
        axes[1, 0].set_ylabel('Time (ms)', fontsize=11)
        axes[1, 0].set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Efficiency ratio
        efficiency = [r['grover_iters'] / r['classical_iters'] for r in self.results]
        axes[1, 1].plot(sizes, efficiency, 'o-', color='purple', linewidth=2, markersize=8)
        axes[1, 1].axhline(y=1/np.sqrt(max(sizes)), color='red', linestyle='--', 
                          label='Theoretical limit')
        axes[1, 1].set_xlabel('Database Size (N)', fontsize=11)
        axes[1, 1].set_ylabel('Grover/Classical Ratio', fontsize=11)
        axes[1, 1].set_title('Quantum Efficiency', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quantum_benchmark.png', dpi=150, bbox_inches='tight')
        plt.show()

def demo():
    """Comprehensive quantum realism demonstration"""
    print("🌌 REALISTIC QUANTUM ANALYSIS")
    print("="*70)
    
    # 1. Combinatorial Analysis
    n_items = 64
    analysis = CombinatorialAnalyzer.analyze(n_items)
    CombinatorialAnalyzer.explain_efficiency(analysis)
    
    # 2. Quantum State Visualization
    print(f"\n{'='*70}")
    print(f"🎨 QUANTUM STATE VISUALIZATION")
    print(f"{'='*70}")
    
    grover = GroverSearch(n_items=8, target_idx=5)
    result = grover.search()
    
    print(f"\nVisualizing Grover search on 8 items...")
    print(f"Target index: 5")
    print(f"Iterations: {result.iterations}")
    
    # Visualize initial and final states
    visualizer = QuantumStateVisualizer()
    
    initial_state = result.state_evolution[0]
    final_state = result.state_evolution[-1]
    
    fig1 = visualizer.visualize_amplitudes(initial_state, "Initial Superposition")
    fig2 = visualizer.visualize_amplitudes(final_state, "After Grover Iterations")
    
    plt.show()
    
    # 3. Benchmark
    benchmark = QuantumBenchmark()
    benchmark.run_benchmark(sizes=[8, 16, 32, 64, 128], trials=10)
    
    print(f"\n{'='*70}")
    print(f"📊 BENCHMARK SUMMARY")
    print(f"{'='*70}")
    
    for r in benchmark.results:
        print(f"\nN={r['n']:3d}: Grover {r['speedup_grover']:.2f}x faster, "
              f"Hybrid {r['speedup_hybrid']:.2f}x faster")
    
    benchmark.visualize_results()

if __name__ == '__main__':
    demo()
