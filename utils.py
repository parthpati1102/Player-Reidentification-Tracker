import cv2
import numpy as np

# Dictionary to store ID: histogram
id_hist_map = {}

# Step 1: Get histogram from a cropped player image
def get_color_histogram(image_crop):
    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

# Step 2: Compare a new histogram to saved ones
def match_histogram(new_hist, threshold=0.7):
    best_id = None
    best_score = 0
    for player_id, hist in id_hist_map.items():
        score = cv2.compareHist(hist, new_hist, cv2.HISTCMP_CORREL)
        if score > threshold and score > best_score:
            best_score = score
            best_id = player_id
    return best_id
