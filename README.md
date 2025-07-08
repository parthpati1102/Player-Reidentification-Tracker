#  Player Re-Identification in Sports Footage

This project is a real-time computer vision system for **player tracking and re-identification** in football matches using video footage. Built using **YOLOv11** for player detection, **SORT** for tracking, and **color histogram** matching for appearance-based re-identification.

> Built as part of an assignment for Liat.ai's AI Intern role

---

## 📽️ Problem Statement

The goal is to simulate a sports analytics pipeline where each player must retain a consistent identity (ID) even if:

- The player **leaves and re-enters** the frame
- The view **changes due to camera angle** (in multi-camera setups)

This is essential in real-world scenarios for creating player heatmaps, pass networks, and match analytics.

---

## Project Structure

```bash
player-reid-liat/
├── main.py                 # Main entry script
├── detect.py               # YOLOv11-based player detection
├── track.py                # SORT tracker integration
├── sort.py                 # SORT algorithm (copied for simplicity)
├── utils.py                # Re-ID using color histograms
├── models/
│   └── yolov11_custom.pt   # YOLOv11 weights (stored separately via Drive)
├── input_videos/
│   └── 15sec_input_720p.mp4 # Test clip (shared via Drive)
├── README.md               # Project documentation
├── report.md               # Summary of approach and outcomes
└── requirements.txt        # All dependencies

How to Run

1. Clone the Repository
git clone https://github.com/your-username/player-reid-liat.git
cd player-reid-liat

2. Create a Virtual Environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # For Windows

3. Install Dependencies
pip install -r requirements.txt

4. Download YOLOv11 Model and Video Model (.pt file):
Download YOLOv11 model weights

Video Input:
Download 15-second input video

Put them into:
player-reid-liat/
├── models/yolov11_custom.pt
├── input_videos/15sec_input_720p.mp4

5. Run the Project
python main.py

