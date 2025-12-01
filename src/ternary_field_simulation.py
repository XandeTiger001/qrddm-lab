import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TernaryFieldSimulator:
    def __init__(self, grid_size=50, core_position=None, k=2.0):
        self.grid_size = grid_size
        self.core_position = core_position or (grid_size // 2, grid_size // 2)
        self.k = k
        
        # Ternary states: -1 (threat), 0 (neutral), 1 (protected)
        self.state = np.zeros((grid_size, grid_size), dtype=int)
        self.g_field = np.ones((grid_size, grid_size))
        self.M_field = np.zeros((grid_size, grid_size))
        self.r_field = self._calculate_distance_field()
        
        self.history = []
    
    def _calculate_distance_field(self):
        """Computes distance of each cell to the core"""
        y, x = np.ogrid[:self.grid_size, :self.grid_size]
        r = np.sqrt((x - self.core_position[0])**2 + (y - self.core_position[1])**2)
        return np.maximum(r, 0.1)
    
    def inject_attack(self, position, intensity=1.0, radius=3):
        """Injects an attack at a given position"""
        y, x = position
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    dist = np.sqrt(dy**2 + dx**2)
                    if dist <= radius:
                        self.state[ny, nx] = -1
                        self.M_field[ny, nx] = intensity * (1 - dist / radius)
    
    def calculate_M(self, x, y):
        """M(r) = α·S + β·F + γ·V + δ·D + ε·C"""
        S = abs(self.state[y, x]) * 0.5  # Intensity
        
        # F: neighbors in state -1
        neighbors = self._count_threat_neighbors(x, y)
        F = neighbors / 8.0
        
        # V: variation (simplified)
        V = 0.0 if len(self.history) == 0 else abs(self.state[y, x] - self.history[-1][y, x]) / 2.0
        
        # D: proximity to core
        D = 1.0 / (1.0 + self.r_field[y, x])
        
        # C: noise (simplified)
        C = self.M_field[y, x] * 0.3
        
        # Balanced coefficients
        M = 1.0 * S + 1.0 * F + 0.8 * V + 1.2 * D + 0.5 * C
        return np.clip(M, 0.01, 10.0)
    
    def _count_threat_neighbors(self, x, y):
        """Counts neighbors in threat state (-1)"""
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                    if self.state[ny, nx] == -1:
                        count += 1
        return count
    
    def calculate_g(self, x, y):
        """g(r) = 1 - k * M(r) / r"""
        M = self.calculate_M(x, y)
        r = self.r_field[y, x]
        g = 1 - (self.k * M) / r
        return g
    
    def evolve_cell(self, x, y):
        """Evolves the cell state based on g(r)"""
        g = self.calculate_g(x, y)
        
        if g > 0.7:
            return 1  # Protected
        elif g > 0.3:
            return 0  # Neutral
        else:
            return -1  # Threat
  
    def step(self):
        """Execute a simulation step"""
        self.history.append(self.state.copy())
        new_state = np.zeros_like(self.state)
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                new_state[y, x] = self.evolve_cell(x, y)
                self.g_field[y, x] = self.calculate_g(x, y)
        
        self.state = new_state
    
    def get_statistics(self):
        """Field Statistics"""
        threats = np.sum(self.state == -1)
        neutral = np.sum(self.state == 0)
        protected = np.sum(self.state == 1)
        total = self.grid_size ** 2
        
        return {
            'threats': threats,
            'neutral': neutral,
            'protected': protected,
            'threat_ratio': threats / total,
            'protected_ratio': protected / total
        }

def visualize_field(simulator, steps=50):
    """Visualize the evolution of the ternary field"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Inject attacks
    simulator.inject_attack((10, 10), intensity=2.0, radius=5)
    simulator.inject_attack((40, 40), intensity=1.5, radius=4)
    
    def update(frame):
        if frame > 0:
            simulator.step()
        
        stats = simulator.get_statistics()
        
        for ax in axes:
            ax.clear()
        
        # Ternary state
        im1 = axes[0].imshow(simulator.state, cmap='RdYlGn', vmin=-1, vmax=1, interpolation='nearest')
        axes[0].plot(simulator.core_position[0], simulator.core_position[1], 'k*', markersize=15)
        axes[0].set_title(f'Ternary State (Step {frame})')
        axes[0].set_xlabel(f'Threats: {stats["threats"]} | Protected: {stats["protected"]}')
        
        # g(r) field
        im2 = axes[1].imshow(simulator.g_field, cmap='coolwarm', vmin=-1, vmax=2, interpolation='bilinear')
        axes[1].plot(simulator.core_position[0], simulator.core_position[1], 'k*', markersize=15)
        axes[1].set_title('g(r) = 1 - kM/r Field')
        
        # Distance to core
        im3 = axes[2].imshow(simulator.r_field, cmap='viridis', interpolation='bilinear')
        axes[2].plot(simulator.core_position[0], simulator.core_position[1], 'r*', markersize=15)
        axes[2].set_title('Distance to Core r')
        
        plt.tight_layout()
        return [im1, im2, im3]
    
    anim = FuncAnimation(fig, update, frames=steps, interval=200, blit=False)
    plt.show()
    
    return anim


def run_simulation():
    print("🌌 Ternary Digital Field Simulation\n")
    print("States: -1 (threat) | 0 (neutral) | 1 (protected)\n")
    print("=" * 60)
    
    simulator = TernaryFieldSimulator(grid_size=50, k=2.0)
    
    # Inject attacks
    simulator.inject_attack((10, 10), intensity=2.0, radius=5)
    simulator.inject_attack((40, 40), intensity=1.5, radius=4)
    
    print("\n📊 Field Evolution:\n")
    
    for step in range(20):
        simulator.step()
        stats = simulator.get_statistics()
        
        if step % 5 == 0:
            print(
                f"Step {step:2d}: Threats={stats['threats']:3d} | "
                f"Neutral={stats['neutral']:3d} | "
                f"Protected={stats['protected']:3d} | "
                f"Threat Ratio={stats['threat_ratio']:.2%}"
            )
    
    print("\n" + "=" * 60)
    print("\n🎯 Final Result:")
    final_stats = simulator.get_statistics()
    print(f"   Threat cells: {final_stats['threats']} ({final_stats['threat_ratio']:.1%})")
    print(f"   Protected cells: {final_stats['protected']} ({final_stats['protected_ratio']:.1%})")
    
    if final_stats['protected_ratio'] > 0.7:
        print("\n   ✅ Stable system - Defense dominant")
    elif final_stats['threat_ratio'] > 0.5:
        print("\n   🚨 Compromised system - Threat dominant")
    else:
        print("\n   ⚠️  Unstable equilibrium system")
    
    return simulator


if __name__ == '__main__':
    simulator = run_simulation()
    
    print("\n🎬 Generating visualization...")
    visualize_field(simulator, steps=30)
