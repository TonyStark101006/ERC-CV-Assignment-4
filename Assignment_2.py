#I tried to update score with no of blocks passing the end of page but I was not successfull till now.  
#The code which I have made according to the template is facing some issues.
#Also, the code which I had made had several more features. But due to bug issues, I haven't uploaded that. I am uploading the base code of the programme which I had made.
#During my interview, I will bring the code as promised. Thak you for taking this for consideration 'u'


import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480

# Player dimensions
PLAYER_WIDTH, PLAYER_HEIGHT = 50, 50

# Falling block dimensions
BLOCK_WIDTH, BLOCK_HEIGHT = 50, 50
BLOCK_SPEED = 5

# Function to draw the player-controlled object
def draw_player(frame, player_pos):
    cv2.rectangle(frame, player_pos, (player_pos[0] + PLAYER_WIDTH, player_pos[1] + PLAYER_HEIGHT), (0, 255, 0), -1)

# Function to draw falling blocks
def draw_blocks(frame, blocks):
    for block in blocks:
        cv2.rectangle(frame, block, (block[0] + BLOCK_WIDTH, block[1] + BLOCK_HEIGHT), (0, 0, 255), -1)

# Initialize player position
player_pos = [SCREEN_WIDTH // 2 - PLAYER_WIDTH // 2, SCREEN_HEIGHT - PLAYER_HEIGHT - 10]

# Initialize falling blocks list
blocks = []

# Initialize game variables
game_over = False
score = 0

# Function to generate random blocks
def generate_block():
    x = random.randint(0, SCREEN_WIDTH - BLOCK_WIDTH)
    y = -BLOCK_HEIGHT  # Start above the screen
    return [x, y]

# Function to check for collisions
def check_collision(player_pos, block):
    px, py = player_pos
    bx, by = block

    # Simple bounding box collision detection
    if (px < bx + BLOCK_WIDTH and px + PLAYER_WIDTH > bx and
        py < by + BLOCK_HEIGHT and py + PLAYER_HEIGHT > by):
        return True
    return False

# Open webcam for capturing video
cap = cv2.VideoCapture(0)
cap.set(3, SCREEN_WIDTH)  # Set width
cap.set(4, SCREEN_HEIGHT)  # Set height

# Main game loop
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror-like effect
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    result = hands.process(frame_rgb)

    # Draw the player-controlled object based on hand position
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get coordinates of the hand's center (index finger tip for control)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            hand_x = int(index_finger_tip.x * SCREEN_WIDTH)
            hand_y = int(index_finger_tip.y * SCREEN_HEIGHT)

            # Update player position (constrain within the screen)
            player_pos[0] = np.clip(hand_x - PLAYER_WIDTH // 2, 0, SCREEN_WIDTH - PLAYER_WIDTH)

            # Draw hand landmarks for debugging
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Generate and move falling blocks
    if random.random() < 0.02:  # Randomly generate new blocks
        blocks.append(generate_block())

    # Move blocks down the screen
    for block in blocks:
        block[1] += BLOCK_SPEED

    # Remove blocks that fall off the screen
    blocks = [block for block in blocks if block[1] < SCREEN_HEIGHT]

    # Check for collisions
    for block in blocks:
        if check_collision(player_pos, block):
            game_over = True
            break

    # Draw the player and blocks
    draw_player(frame, player_pos)
    draw_blocks(frame, blocks)

    # Display the score
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Increment score (based on time survived)
    score += 1

    # Show the frame
    cv2.imshow('Hand Tracking Game', frame)

    # Check for game over
    if game_over:
        cv2.putText(frame, "Game Over!", (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow('Hand Tracking Game', frame)
        cv2.waitKey(3000)  # Display Game Over for 3 seconds
        break

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()