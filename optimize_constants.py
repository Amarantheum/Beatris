import numpy as np
import time
import cv2
from mss import mss
import os
from tkinter import Tk
import tetris_logic
import subprocess
import argparse
from pynput.keyboard import Key, Controller as kController

keyboardCont = kController()

parser = argparse.ArgumentParser(description="Optimize some constants.")

# Arguments
parser.add_argument("constant", choices = ["hole_cost", "hole_height_cost", "height_cost", "jagged_cost", "line_value", "height_threshold"])
parser.add_argument("--low", type = int, help = "Lower bound of optimization", default = 10)
parser.add_argument("--high", type = int, help = "Upper bound of optimization", default = 10)
parser.add_argument("--trials", type = int, help = "Number of trials per constant candidate.", default = 1)
args = parser.parse_args()

# Evaluation: Nested For Loop Brrrrrr
best_time = [0, None] # best_time: [time, constant value]
for n in range(args.low, args.high + 1):
    total_survival_time = 0
    for trial in range(args.trials):
        # Determine Constant, Reset Value
        if args.constant == "hole_cost":
            tetris_logic.set_hole_cost(n)
        elif args.constant == "hole_height_cost":
            tetris_logic.set_hole_height_cost(n)
        elif args.constant == "height_cost":
            tetris_logic.set_height_cost(n)
        elif args.constant == "jagged_cost":
            tetris_logic.set_jagged_cost(n)
        elif args.constant == "line_value":
            tetris_logic.set_line_value(n)
        elif args.constant == "height_threshold":
            tetris_logic.set_height_threshold(n)
        else:
            sys.exit("What the nani this isn't even possible")
        # Evaluation
        total_survival_time = 0
        start_time = time.time()
        subprocess.call(["python3", "jstris.py"])
        survival_time = time.time() - start_time
        total_survival_time += survival_time
        # Press keyboard shortcut for restarting game and repeat # of trials time
        keyboardCont.press("j") # IMPORTANT: I set the keybind for restart game to "j", because f4 was trolling me on Mac
        keyboardCont.release("j")
        time.sleep(0.05)
    if total_survival_time/args.trials > best_time[0]:
        best_time[0] = total_survival_time/args.trials
        best_time[1] = n
print("Best average time: " + str(best_time[0]))
print("Best value for " + args.constant + " between " + args.low + " and " + args.high + " is: " + str(best_time[1]))
