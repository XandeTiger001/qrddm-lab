#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from action_limits import demo_tiered_limits
from integrated_defense import demo_integrated_defense

def main():
    print("Cyber Event Horizon - Action Limits Demo\n")
    print("Testing tiered defense with graceful degradation...")
    print("Policy: MONITOR -> THROTTLE -> REDIRECT -> BLOCK\n")
    
    # Demo 1: Basic tiered limits
    demo_tiered_limits()
    
    print("\n" + "="*60 + "\n")
    
    # Demo 2: Integrated defense system
    demo_integrated_defense()
    
    print("\nDemo completed successfully!")
    print("\nKey Features Demonstrated:")
    print("- Progressive escalation (MONITOR -> THROTTLE -> REDIRECT -> BLOCK)")
    print("- Honeypot redirection before blocking")
    print("- Violation history tracking with escalation")
    print("- Integration with quantum defense layers")
    print("- Non-destructive actions prioritized")
    print("- Graceful degradation under attack")

if __name__ == '__main__':
    main()