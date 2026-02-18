#!/usr/bin/env python3
"""
Quick test of Stockfish opponent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rl.env import ChessEnv
from rl.opponents import StockfishOpponent, RandomOpponent

def test_stockfish():
    """Test Stockfish opponent plays moves."""
    print("=" * 80)
    print("Testing Stockfish Opponent")
    print("=" * 80)
    
    # Create environment and opponents
    env = ChessEnv()
    sf = StockfishOpponent(depth=5, name="Stockfish(d=5)")
    random_opp = RandomOpponent()
    
    print(f"\nOpponents: {sf.get_name()} vs {random_opp.get_name()}")
    print("Running 3 games...")
    
    for game_num in range(3):
        print(f"\n[Game {game_num + 1}]")
        
        obs, info = env.reset()
        done = False
        moves = 0
        
        while not done:
            # Stockfish (white)
            action_sf = sf.get_action(env.board)
            obs, reward, done, truncated, info = env.step(action_sf)
            moves += 1
            
            if done or truncated:
                break
            
            # Random (black)
            action_rand = random_opp.get_action(env.board)
            obs, reward, done, truncated, info = env.step(action_rand)
            moves += 1
        
        result = info.get('result', 'unknown')
        print(f"  Result: {result} ({moves} moves)")
    
    print("\n" + "=" * 80)
    print("✓ Stockfish opponent works!")
    print("=" * 80)

if __name__ == '__main__':
    try:
        test_stockfish()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
