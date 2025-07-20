
# AI Camera Streaming Dashboard 🚀

An advanced real-time camera streaming system with AI-powered object detection and face recognition, built using FastAPI, OpenCV, and YOLOv8. Includes a responsive dashboard UI for managing multiple cameras.

---

## 🔧 Features

- 📹 Real-time video streaming from RTSP cameras
- 🧠 AI detection with YOLOv8 (person detection)
- 🧍‍♂️ Face recognition for known individuals
- 🖥️ Grid and single view camera layout
- ⚙️ Add, update, delete, and restart camera streams
- ✅ AI toggle on/off
- 📊 Dashboard with live stats and camera control

---

## 📁 Project Structure

```text
.
├── try.py               # FastAPI backend with AI and camera streaming logic
├── dashboard.html       # Frontend dashboard UI
├── known_faces/         # Directory to store face images for recognition
├── templates/           # Jinja2 template directory (should contain dashboard.html)
├── static/              # (Optional) for CSS/JS files if split from HTML
