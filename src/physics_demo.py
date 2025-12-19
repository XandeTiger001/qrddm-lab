import numpy as np
from adaptive_mass import BlackHoleEntropy, InformationDecay, AdaptiveCurvatureTensor

def demo_physics_tools():
    print("🌌 Physics-Inspired AI Tools Demo\n")
    
    # Test data - simulated attack reconstruction
    original_data = np.array([1.0, 0.8, 0.6, 0.9, 0.7, 0.5, 0.8, 0.6])
    corrupted_data = np.array([0.9, 0.7, 0.5, 0.8, 0.6, 0.4, 0.7, 0.5])
    
    # 1. Black Hole Entropy - Complexity adjusts confidence
    print("🌀 BLACK HOLE ENTROPY")
    entropy_calc = BlackHoleEntropy()
    entropy = entropy_calc.calculate_scene_entropy(corrupted_data)
    adjusted_confidence = entropy_calc.adjust_confidence_by_entropy(0.85, entropy)
    
    print(f"   Scene Complexity (Entropy): {entropy:.3f} bits")
    print(f"   Confidence: 0.850 → {adjusted_confidence:.3f}")
    print(f"   Uncertainty Penalty: {((0.85 - adjusted_confidence) / 0.85) * 100:.1f}%")
    
    # 2. Information Decay - Data crossing event horizon
    print("\n📉 INFORMATION DECAY")
    decay_sim = InformationDecay(decay_rate=0.4)
    decay_stages = decay_sim.simulate_horizon_crossing(original_data[:4], steps=4)
    
    for i, stage in enumerate(decay_stages):
        status = "✅ Recoverable" if stage['recoverable'] else "❌ Lost"
        print(f"   Stage {i+1}: Integrity {stage['data_integrity']:.1%} - {status}")
    
    # 3. Adaptive Curvature - Tensor warps AI perception
    print("\n🌐 ADAPTIVE CURVATURE TENSOR")
    tensor_calc = AdaptiveCurvatureTensor(tensor_size=8)
    
    # Split data into regions (low vs high information)
    low_info_region = np.array([0.1, 0.1, 0.2, 0.1])  # Low variance = low info
    high_info_region = np.array([0.9, 0.2, 0.8, 0.3])  # High variance = high info
    
    curvature_tensor = tensor_calc.adapt_to_information_landscape([low_info_region, high_info_region])
    warped_perception = tensor_calc.warp_ai_perception(original_data, curvature_tensor)
    
    print(f"   Curvature Strength: {np.trace(curvature_tensor):.3f}")
    print(f"   Original Data: {original_data[:4]}")
    print(f"   Warped Perception: {warped_perception[:4]}")
    print(f"   Relativistic Shift: {np.linalg.norm(warped_perception - original_data):.3f}")
    
    # Combined Effect
    print(f"\n🎯 COMBINED PHYSICS EFFECT")
    print(f"   Entropy reduces confidence by {((0.85 - adjusted_confidence) / 0.85) * 100:.1f}%")
    print(f"   Information decay: {decay_stages[-1]['data_integrity']:.1%} integrity remaining")
    print(f"   Curvature warps perception by {np.linalg.norm(warped_perception - original_data):.3f} units")
    
    # Elegant physics principle
    final_uncertainty = (1.0 - adjusted_confidence) + (1.0 - decay_stages[-1]['data_integrity']) + (np.linalg.norm(warped_perception - original_data) / 10)
    print(f"   Total Uncertainty: {final_uncertainty:.3f} (higher complexity = greater uncertainty)")

if __name__ == '__main__':
    demo_physics_tools()