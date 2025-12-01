import numpy as np
import matplotlib.pyplot as plt
from src.schwarzschild_defense import SchwarzschildDefense

def visualize_metric():
    defense = SchwarzschildDefense(D=1.5, c=1.0)
    
    # Different intensities of attack
    M_values = [0.5, 1.0, 2.0, 5.0]
    r = np.linspace(0.1, 10, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Schwarzschild Metric - Cyber defense field', fontsize=16)
    
    # Φ(r) - Stability Indicator
    ax1 = axes[0, 0]
    for M in M_values:
        phi = [defense.phi(ri, M) for ri in r]
        ax1.plot(r, phi, label=f'M={M}')
    ax1.axhline(y=defense.T_shadow, color='g', linestyle='--', label='Safe Threshold')
    ax1.axhline(y=defense.T_crit, color='r', linestyle='--', label='Critical Threshold')
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Distancy r (To the nucleus)')
    ax1.set_ylabel('Φ(r)')
    ax1.set_title('Stability Indicator')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Curvature C(r)
    ax2 = axes[0, 1]
    for M in M_values:
        C = [defense.curvature(ri, M) for ri in r]
        ax2.plot(r, C, label=f'M={M}')
    ax2.set_xlabel('Distancy r')
    ax2.set_ylabel('C(r)')
    ax2.set_title('Defense Field Curvature')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Schwarzschild ray
    ax3 = axes[1, 0]
    M_range = np.linspace(0.1, 10, 100)
    r_s = [defense.schwarzschild_radius(M) for M in M_range]
    ax3.plot(M_range, r_s, 'r-', linewidth=2)
    ax3.fill_between(M_range, 0, r_s, alpha=0.3, color='red', label='Horizon zone')
    ax3.set_xlabel('Attack Energy M')
    ax3.set_ylabel('Schwarzschild Ray r_s')
    ax3.set_title('Event Horizon')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Classification Map
    ax4 = axes[1, 1]
    r_grid = np.linspace(0.1, 10, 100)
    M_grid = np.linspace(0.1, 10, 100)
    R, M = np.meshgrid(r_grid, M_grid)
    
    PHI = 1 - (2 * defense.D * M) / (defense.c**2 * R)
    
    levels = [defense.T_crit, defense.T_shadow]
    contour = ax4.contourf(R, M, PHI, levels=[-1, defense.T_crit, defense.T_shadow, 2], 
                           colors=['red', 'yellow', 'green'], alpha=0.6)
    ax4.contour(R, M, PHI, levels=levels, colors='black', linewidths=1)
    ax4.set_xlabel('Distance r')
    ax4.set_ylabel('Energy M')
    ax4.set_title('Threat classication map')
    
    # Custom Caption
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.6, label='CRITICAL'),
        Patch(facecolor='yellow', alpha=0.6, label='MONITOR'),
        Patch(facecolor='green', alpha=0.6, label='SAFE')
    ]
    ax4.legend(handles=legend_elements, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('schwarzschild_defense_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ Saved view: schwarzschild_defense_visualization.png")
    plt.show()

if __name__ == '__main__':
    visualize_metric()
