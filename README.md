# Face-Recognition-Attendance-System
I developed a Face Recognition Attendance System using Python, Flask, and OpenCV to automate attendance tracking. The system opens a live camera feed, detects faces in real-time, and compares them with stored images to verify identity. When a match is found, the system marks the person as present and updates the attendance record, which is then displayed on a web page. It's designed to be fast and efficient, eliminating the need for manual attendance while ensuring accuracy through facial recognition technology.

---

# 🧠 Real-Time Face Recognition System with Stabilization

A robust, real-time face recognition system built using **Python**, combining the power of **OpenCV** for video capture and detection, **DeepFace (FaceNet)** for deep learning-based facial recognition, and a custom **FaceTracker** for smoothing and stabilization of detections across frames.

This system not only recognizes known faces but also **learns new ones interactively**, and saves them for future sessions. It is optimized for accuracy, stability, and user-friendly performance, making it ideal for personal or small-scale security and recognition systems.

---

## 🚀 Features

* **📷 Real-Time Face Detection**

  * Uses OpenCV’s Haar Cascades with tuned parameters for initial face localization.

* **📌 Smart Face Tracking & Stabilization**

  * Custom `FaceTracker` class tracks faces across frames, smoothing their movement and reducing jitter using exponential averaging and confidence accumulation.

* **🧠 Deep Learning-Based Recognition**

  * Integrates DeepFace (using FaceNet backend) to convert faces into 128-dimensional embeddings for precise comparison using cosine similarity.

* **🧑‍💻 Interactive Face Enrollment**

  * Detects new, unknown but stable faces and prompts the user via a Tkinter dialog to label them—allowing the system to learn dynamically.

* **💾 Persistent Memory**

  * Saves known face embeddings (`face_data.pkl`) and associated names (`face_names.json`) across runs using `pickle` and `json`.

* **⚡ Recognition Caching**

  * Temporarily stores recognition results for stable tracked faces to avoid redundant computation and improve frame processing speed.

* **📊 Real-Time Visual Feedback**

  * Draws bounding boxes, confidence levels, names, and stability scores on the webcam feed in real-time.

---

## 🔧 Prerequisites

Ensure the following packages are installed:

```bash
pip install opencv-python deepface tensorflow scikit-learn
```

* Python ≥ 3.7
* TensorFlow (GPU recommended for better performance)
* Tkinter (pre-installed with most Python versions)

---

## ▶️ How to Run

1. **Save the Script**
   Save the main Python code as `face_recognition_system.py`.

2. **Launch the App**

   ```bash
   python face_recognition_system.py
   ```

3. **Interact with the System**

   * The webcam feed opens.
   * Detected faces are tracked and analyzed.
   * Unknown but stable faces trigger a popup asking for a name.
   * Recognized faces are labeled in real-time.
   * Press `q` to exit the application.

---

## 📁 Project Structure and Main Components

### ✅ FaceTracker Class

Tracks and stabilizes bounding boxes across frames:

* **Smoothing**: Uses exponential averaging to reduce jitter.
* **Stability Check**: Requires a face to appear for `n` frames before considering it "stable".
* **Face Association**: Uses IOU (Intersection Over Union) to match new detections with tracked faces.
* **processed\_unknown\_faces**: Prevents duplicate prompts for the same unidentified face.

### ✅ FaceRecognitionSystem Class

Main system logic:

* **Initialization**: Loads saved face data and sets up camera and detection.
* **Face Detection**: Uses `detect_faces_stable()` with preprocessing (equalization, blur, size/aspect ratio filters).
* **Face Encoding**: Uses DeepFace to get embeddings.
* **Recognition**: Uses cosine similarity for comparison and decision-making.
* **Interactive Learning**: Prompts user with `ask_for_name()` to label unknowns.
* **Caching**: Stores and reuses recent recognition results.
* **Main Loop** (`run_recognition()`):

  * Detects faces, updates tracker.
  * Recognizes stable faces.
  * Handles new faces.
  * Renders video with visual annotations.

---

## ⚙️ Customization Options

### FaceTracker Parameters:

| Parameter               | Default | Description                                       |
| ----------------------- | ------- | ------------------------------------------------- |
| `smoothing_factor`      | 0.8     | Smooths bounding boxes; higher = smoother         |
| `min_confidence_frames` | 3       | Minimum frames before a face is considered stable |

### FaceRecognitionSystem Parameters:

| Parameter              | Default  | Description                                                |
| ---------------------- | -------- | ---------------------------------------------------------- |
| `confidence_threshold` | 0.7      | Recognition threshold based on cosine similarity           |
| `cache_timeout`        | 3 sec    | Duration to cache recognition results                      |
| `recognition_interval` | 8 frames | Frequency of running DeepFace recognition                  |
| `detectMultiScale`     | tunable  | Adjust `scaleFactor`, `minSize`, etc. for your environment |

---

## 🧪 Troubleshooting

| Issue                       | Fix                                                                  |
| --------------------------- | -------------------------------------------------------------------- |
| `Error loading face data`   | Safe to ignore on first run. Files are created after first save.     |
| Tkinter dialog doesn’t show | Ensure the UI runs in the main thread (already handled)              |
| Slow processing             | Enable GPU for TensorFlow, or increase `recognition_interval`        |
| Poor recognition/detection  | Improve lighting, tune Haar cascade, or lower `confidence_threshold` |

---

## 💡 Tips for Best Results

* Use well-lit environments.
* Ensure faces are clear and large enough during recognition or enrollment.
* Keep the camera steady for better tracking accuracy.
* Label only faces that are frontal and unobstructed.

---

## 📂 Suggested Repository Layout

```
face-recognition-system/
│
├── face_recognition_system.py       # Main Python script
├── face_data.pkl                    # Saved face encodings (auto-created)
├── face_names.json                  # Saved names of faces (auto-created)
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
```

## Requirements.txt

opencv-python
deepface
tensorflow
scikit-learn
numpy
pillow
