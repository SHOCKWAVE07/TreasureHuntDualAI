from environment.constants import GRID_SIZE  
import os
from datetime import datetime

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def is_valid_move(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def log_game_stats(env, winner, turn_count):
    """
    Logs game statistics into a log file.
    """
    os.makedirs("logs", exist_ok=True)
    with open("logs/game_stats.txt", "a") as log_file:
        log_file.write(f"Game played on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total Turns: {turn_count}\n")
        log_file.write(f"Final Scores: {env.scores}\n")
        log_file.write(f"Winner: {winner}\n")
        log_file.write("-" * 40 + "\n")
