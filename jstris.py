import numpy as np
from pynput.keyboard import Key, Controller as kController
from pynput import keyboard
import time
import cv2
from mss import mss
import os
from tkinter import Tk
import tetris_logic

#print(tetris_logic.mult_py(np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]])))

monitor = Tk()

#time.sleep(3) #For testing 

tetris_logic.set_line_value(0, -100)
tetris_logic.set_line_value(1, -30)
tetris_logic.set_line_value(2, 0)
tetris_logic.set_line_value(3, 1600)

#global variables
box_unit = 24
stored_piece = True
depth = 3
keyboard_delay = 0.016000001
garbage = False

if garbage == True: 
    tetris_logic.set_garbage(True)

#define pieces:
#   0: I block (cyan), 1: O block (yellow), 2: T block (pink), 3: S block (green), 
#   4: Z block (red), 5: J block (blue), 6: L block (orang)
I_bgr = [215, 155,  15]
O_bgr = [2,159,227]
T_bgr = [138, 41, 175]
S_bgr = [1, 177, 89]
Z_bgr = [55, 15, 215]
J_bgr = [198, 65, 33]
L_bgr = [2, 91, 227]
Blank_BGR = [0, 0, 0]
piece_bgr_values = [I_bgr, O_bgr, T_bgr, S_bgr, Z_bgr, J_bgr, L_bgr]
upcoming_pieces = [-1] * 5
saved_piece = -1

#colors:
gray_bgr = [106, 106, 106]
black_bgr = [0, 0, 0]
message_box_bgr = [28, 28, 28]

screen_width = monitor.winfo_screenwidth()
screen_height = monitor.winfo_screenheight()

bounding_box = {'top': 0, 'left': 0, 'width': screen_width, 'height': screen_height}
sct = mss()

game_over = False

keyboardCont = kController()

fullscr_img = cv2.cvtColor(np.array(sct.grab(bounding_box)), cv2.COLOR_BGRA2BGR)
#cv2.imwrite("test.png", fullscr_img)

corner_img = cv2.imread(os.path.join("Res", "corner.png"), cv2.IMREAD_COLOR)

gg_img = cv2.imread(os.path.join("Res", "gg.png"), cv2.IMREAD_COLOR)
gg_alt_img = cv2.imread(os.path.join("Res", "gg_alt.png"), cv2.IMREAD_COLOR)

corner_coords = tetris_logic.get_corner_coords(fullscr_img, corner_img, box_unit)
print(corner_coords)

game_box = {'top': corner_coords[0], 'left': corner_coords[1], 'width': box_unit * 14, 'height': box_unit * 20}

game_img = cv2.cvtColor(np.array(sct.grab(game_box)), cv2.COLOR_BGRA2BGR)

upcoming_pieces = [0, 0, 0, 0, 0]
 
#checks the top level for any gray squares frite
def check_game_over():
    global game_over
    global corner_coords
    global game_img
    for square in game_img[round(box_unit/2), round(box_unit/2): box_unit * 10 + round(box_unit/2): box_unit]:
        if array_diff(square, gray_bgr) == 0:
            print("GG took an L")
            game_over = True
            return True
    game_over = False
    return False

def check_game_starting():
    global message_box_bgr
    global box_unit
    global game_img
    if array_diff(game_img[box_unit * 12, 0], message_box_bgr) == 0:
        return True
    else:
        return False


def array_diff(arr1, arr2):
    if len(arr1) != len(arr2):
        raise ValueError("Smh. Arrays must be the same size")
    Error = 0
    for n in range(len(arr1)):
        Error += (arr1[n] - arr2[n])*2
    return Error/len(arr1)

# The Method that was not Used
def get_upcoming_pieces():
    pieces = []
    for i in range(5):
        piece = tetris_logic.identify_piece(fullscr_img[corner_coords[1] + round(box_unit * (1.5 + i * 3)), corner_coords[0] + round(box_unit * 12.5)])
        if piece == 404:
            piece = tetris_logic.identify_piece(fullscr_img[corner_coords[1] + round(box_unit * (2.5 + i * 3)), corner_coords[0] + round(box_unit * 12.5)])
            if piece == 404:
                raise RuntimeError("Smh. Color wrong or grabbing from wrong place")
            else:
                pieces.append(piece)
        else:
            pieces.append(piece)
    return pieces

#keyboard_delay = 0.5
def make_move(move):
    if move[2] == True:
        keyboardCont.press('c')
        keyboardCont.release('c')
        time.sleep(keyboard_delay)
    for i in range(0, move[0]):
        keyboardCont.press(Key.up)
        keyboardCont.release(Key.up)
        time.sleep(keyboard_delay)
    if move[1] < 0:
        for i in range(0, abs(move[1])):
            keyboardCont.press(Key.left)
            keyboardCont.release(Key.left)
            time.sleep(keyboard_delay)
    else:
        for i in range(0, abs(move[1])):
            keyboardCont.press(Key.right)
            keyboardCont.release(Key.right)
            time.sleep(keyboard_delay)
    keyboardCont.press(Key.space)
    keyboardCont.release(Key.space)
    time.sleep(keyboard_delay)

def get_falling_piece():
    global game_img
    global box_unit
    piece = tetris_logic.identify_piece(game_img[0, box_unit * 5])
    if piece != 404:
        return piece

#make new pic
def update_pic():
    global bounding_box
    global keyboardCont
    global game_img
    global upcoming_pieces
    global saved_piece
    global depth
    global stored_piece
    game_img = cv2.cvtColor(np.array(sct.grab(game_box)), cv2.COLOR_BGRA2BGR)
    move = tetris_logic.get_next_move(game_img, depth, stored_piece)
    make_move(move)
    time.sleep(0)
#This checks if the game has started at all, or if the previous game didn't end
while check_game_starting() == False:
    game_img = cv2.cvtColor(np.array(sct.grab(game_box)), cv2.COLOR_BGRA2BGR)

#If the game is starting (display ready/go banner), takes screenshot of all upcoming pieces
print("Game Starting...")
#game_img = cv2.cvtColor(np.array(sct.grab(game_box)), cv2.COLOR_BGRA2BGR)
#upcoming_pieces = tetris_logic.get_upcoming_pieces(fullscr_img, box_unit, corner_coords)

while check_game_starting():
    game_img = cv2.cvtColor(np.array(sct.grab(game_box)), cv2.COLOR_BGRA2BGR)
print("Game Start!")

while get_falling_piece() == None:
    game_img = cv2.cvtColor(np.array(sct.grab(game_box)), cv2.COLOR_BGRA2BGR)



if stored_piece:
    tetris_logic.set_stored_piece(get_falling_piece())
    upcoming_pieces = tetris_logic.get_upcoming_pieces(game_img, box_unit)

else:
    upcoming_pieces = [get_falling_piece()] + tetris_logic.get_upcoming_pieces(game_img, box_unit)
tetris_logic.set_upcoming_pieces(upcoming_pieces)

if stored_piece:
    keyboardCont.press('c')
    keyboardCont.release('c')
    time.sleep(keyboard_delay)
    game_img = cv2.cvtColor(np.array(sct.grab(game_box)), cv2.COLOR_BGRA2BGR)
    move = tetris_logic.get_next_move(game_img, 1, False)
    make_move(move)
#time.sleep(0)

print(upcoming_pieces)

while check_game_over() == False:
    update_pic()
    



#heights = [0] * 10

#next box 28 from board

#keyboardCont.press('a')
#keyboardCont.release('a')
