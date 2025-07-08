import cv2
from detect import PlayerDetector
from track import PlayerTracker
from utils import get_color_histogram, match_histogram, id_hist_map

# Initialize paths
video_path = 'input_videos/15sec_input_720p.mp4'
model_path = 'models/yolov11_custom.pt'

# Initialize detector and tracker
detector = PlayerDetector(model_path)
tracker = PlayerTracker()

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracked_objects = tracker.track(detections)

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)

        # Crop player from frame
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Get histogram
        new_hist = get_color_histogram(crop)

        # If it's a new ID, check if it matches an old one
        if track_id not in id_hist_map:
            matched_id = match_histogram(new_hist)
            if matched_id is not None:
                # Reassign old ID
                track_id = matched_id

        # Update histogram map
        id_hist_map[track_id] = new_hist

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show the frame after drawing all players
    cv2.imshow("Player Tracking", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
