

import os
import cv2
import time
import asyncio
import threading
import queue
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from collections import deque
import logging
import face_recognition
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from supabase import create_client, Client
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress ultralytics verbose output
logging.getLogger('ultralytics').setLevel(logging.INFO)

# Supabase configuration
SUPABASE_URL = "https://yjpdozcecvxnqukxenmw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlqcGRvemNlY3Z4bnF1a3hlbm13Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDczNzQ5MTMsImV4cCI6MjA2Mjk1MDkxM30.87wH93kLA9u1sye9W9RRN2MosKoR70Segyu5SiiK4Tg"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Global variables
active_streams: Dict[str, 'CameraStream'] = {}
ai_detection_enabled = False
grid_cameras: List[str] = []
known_face_encodings = []
known_face_names = []

# Default cameras configuration
DEFAULT_CAMERAS = [
    {
        "cam_id": "cam_01",
        "cam_location": "Office Main Gate",
        "cam_url": "rtsp://admin:password@port",
        "cam_password": "password"
    }
]

# Pydantic models
class Camera(BaseModel):
    cam_id: str
    cam_location: str
    cam_url: str
    cam_password: str

class CameraUpdate(BaseModel):
    cam_location: Optional[str] = None
    cam_url: Optional[str] = None
    cam_password: Optional[str] = None

class AIToggle(BaseModel):
    enabled: bool

class GridToggle(BaseModel):
    camera_ids: List[str]

class AIProcessor:
    """Handles AI processing including object detection and face recognition"""
    def __init__(self):
        self.yolo_model = None
        self.model_loaded = False
        self.load_models()
        self.load_known_faces()
    
    def load_models(self):
        """Load YOLO model with better error handling"""
        try:
            # Try to find YOLO model in current directory
            possible_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
            model_path = None
            
            for model_file in possible_models:
                if os.path.exists(model_file):
                    model_path = model_file
                    break
            
            if model_path:
                logger.info(f"Loading YOLO model: {model_path}")
                self.yolo_model = YOLO(model_path)
                self.model_loaded = True
                logger.info(f"YOLO model loaded successfully from {model_path}")
            else:
                # Try to download yolov8n.pt automatically
                logger.info("No local YOLO model found, downloading yolov8n.pt...")
                self.yolo_model = YOLO('yolov8n.pt')  # This will download automatically
                self.model_loaded = True
                logger.info("YOLO model downloaded and loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.yolo_model = None
            self.model_loaded = False
    
    def load_known_faces(self, known_faces_dir='known_faces'):
        """Load known face images and encodings"""
        global known_face_encodings, known_face_names
        
        known_face_encodings.clear()
        known_face_names.clear()
        
        if not os.path.exists(known_faces_dir):
            os.makedirs(known_faces_dir)
            logger.info(f"Created directory {known_faces_dir} - please add face images")
            return
            
        face_count = 0
        for filename in os.listdir(known_faces_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
                try:
                    image_path = os.path.join(known_faces_dir, filename)
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        known_face_encodings.append(face_encodings[0])
                        name = os.path.splitext(filename)[0]
                        known_face_names.append(name)
                        face_count += 1
                        logger.info(f"Loaded face: {name}")
                    else:
                        logger.warning(f"No face found in {filename}")
                        
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
        
        logger.info(f"Loaded {face_count} known faces")
    
    def process_frame(self, frame):
        """Process frame with object detection and face recognition"""
        if not ai_detection_enabled:
            return frame
            
        if not self.model_loaded or self.yolo_model is None:
            # Draw warning on frame
            cv2.putText(frame, "AI Model Not Loaded", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        try:
            # Create a copy to avoid modifying original
            processed_frame = frame.copy()
            
            # Run YOLO object detection (only person class - class 0)
            results = self.yolo_model(processed_frame, classes=[0], verbose=True, conf=0.5)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Draw detection box
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Person {confidence:.2f}"
                        
                        # Draw label background
                        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(processed_frame, (x1, y1 - label_height - 10), 
                                    (x1 + label_width, y1), (0, 255, 0), -1)
                        cv2.putText(processed_frame, label, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        
                        # Face recognition within person box (if we have known faces)
                        if known_face_encodings and len(known_face_encodings) > 0:
                            # Extract person ROI with some padding
                            roi_x1 = max(0, x1)
                            roi_y1 = max(0, y1)
                            roi_x2 = min(processed_frame.shape[1], x2)
                            roi_y2 = min(processed_frame.shape[0], y2)
                            
                            person_roi = processed_frame[roi_y1:roi_y2, roi_x1:roi_x2]
                            
                            if person_roi.size > 0 and person_roi.shape[0] > 50 and person_roi.shape[1] > 50:
                                try:
                                    # Convert to RGB for face_recognition
                                    rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                                    
                                    # Find faces in the ROI
                                    face_locations = face_recognition.face_locations(rgb_roi, model='hog')
                                    
                                    if face_locations:
                                        face_encodings = face_recognition.face_encodings(rgb_roi, face_locations)
                                        
                                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                                            # Compare with known faces
                                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                                            name = "Unknown"
                                            
                                            # Find best match
                                            if True in matches:
                                                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                                best_match_index = np.argmin(face_distances)
                                                if matches[best_match_index]:
                                                    name = known_face_names[best_match_index]
                                            
                                            # Draw face rectangle (adjust coordinates back to full frame)
                                            face_x1 = roi_x1 + left
                                            face_y1 = roi_y1 + top
                                            face_x2 = roi_x1 + right
                                            face_y2 = roi_y1 + bottom
                                            
                                            # Draw face box
                                            cv2.rectangle(processed_frame, (face_x1, face_y1), (face_x2, face_y2), (0, 0, 255), 2)
                                            
                                            # Draw name background
                                            (name_width, name_height), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                                            cv2.rectangle(processed_frame, (face_x1, face_y2), 
                                                        (face_x1 + name_width + 10, face_y2 + name_height + 10), 
                                                        (0, 0, 255), -1)
                                            cv2.putText(processed_frame, name, (face_x1 + 5, face_y2 + name_height + 5),
                                                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                                                       
                                except Exception as face_error:
                                    logger.debug(f"Face recognition error: {face_error}")
            
            # Add AI status indicator
            status_text = f"AI: ON"
            cv2.putText(processed_frame, status_text, (10, processed_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"AI processing error: {e}")
            # Draw error on frame
            cv2.putText(frame, f"AI Error: {str(e)[:50]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame

# Global AI processor
ai_processor = AIProcessor()

class CameraStream:
    """Optimized camera stream handler with HEVC error fixes"""
    def __init__(self, cam_id: str, cam_url: str):
        self.cam_id = cam_id
        self.cam_url = cam_url
        self.capture = None
        self.frame_queue = deque(maxlen=3)  # Smaller buffer to reduce latency
        self.running = False
        self.thread = None
        self.last_frame = None
        self.frame_lock = threading.Lock()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.fps = 15  # Reduced FPS for stability
        self.frame_skip = 2  # Skip more frames
        self.frame_count = 0
        self.status = "Initializing"
        self.last_update_time = time.time()
        
    def start(self):
        """Start the camera stream thread"""
        if self.running:
            return True
            
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started camera stream: {self.cam_id}")
        return True
    
    def _init_capture(self):
        """Initialize video capture with HEVC error handling"""
        # Try different backend combinations to handle HEVC streams
        backend_configs = [
            (cv2.CAP_FFMPEG, {
                cv2.CAP_PROP_BUFFERSIZE: 1,
                cv2.CAP_PROP_FPS: 15,
                cv2.CAP_PROP_FRAME_WIDTH: 1280,
                cv2.CAP_PROP_FRAME_HEIGHT: 720,
                # Add codec preferences to handle HEVC better
                cv2.CAP_PROP_FOURCC: cv2.VideoWriter_fourcc('H', '2', '6', '4')
            }),
            (cv2.CAP_GSTREAMER, {
                cv2.CAP_PROP_BUFFERSIZE: 1,
                cv2.CAP_PROP_FPS: 15,
            }),
            (cv2.CAP_ANY, {
                cv2.CAP_PROP_BUFFERSIZE: 1,
                cv2.CAP_PROP_FPS: 15,
            })
        ]
        
        for backend, properties in backend_configs:
            try:
                logger.info(f"Trying backend {backend} for {self.cam_id}")
                cap = cv2.VideoCapture(self.cam_url, backend)
                
                if cap.isOpened():
                    # Set properties
                    for prop, value in properties.items():
                        cap.set(prop, value)
                    
                    # Test frame read with timeout handling
                    start_time = time.time()
                    ret, frame = cap.read()
                    read_time = time.time() - start_time
                    
                    if ret and frame is not None and read_time < 5.0:  # 5 second timeout
                        logger.info(f"Successfully connected to {self.cam_id} using backend {backend}")
                        self.status = "Connected"
                        self.reconnect_attempts = 0
                        return cap
                    else:
                        logger.warning(f"Backend {backend} opened but failed to read frame or timed out")
                        cap.release()
                else:
                    logger.warning(f"Backend {backend} failed to open stream")
                    
            except Exception as e:
                logger.debug(f"Backend {backend} failed with error: {e}")
        
        self.status = "Connection failed"
        return None
    
    def _capture_loop(self):
        """Main capture loop with better error handling"""
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.running:
            try:
                if self.capture is None or not self.capture.isOpened():
                    if not self._reconnect():
                        time.sleep(2)
                        continue
                
                # Non-blocking read with timeout
                ret, frame = self.capture.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Too many consecutive failures for {self.cam_id}, reconnecting...")
                        if self.capture:
                            self.capture.release()
                            self.capture = None
                        consecutive_failures = 0
                    time.sleep(0.1)
                    continue
                
                consecutive_failures = 0  # Reset failure counter
                self.frame_count += 1
                
                # Skip frames for performance
                if self.frame_count % self.frame_skip != 0:
                    continue
                
                # Resize frame
                if frame.shape[:2] != (720, 1280):
                    frame = cv2.resize(frame, (1280, 720))
                
                # Store frame safely
                with self.frame_lock:
                    self.last_frame = frame.copy()
                    if len(self.frame_queue) >= self.frame_queue.maxlen:
                        self.frame_queue.popleft()
                    self.frame_queue.append(frame.copy())
                    self.last_update_time = time.time()
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                logger.error(f"Capture error for {self.cam_id}: {e}")
                consecutive_failures += 1
                time.sleep(0.5)
    
    def _reconnect(self):
        """Attempt to reconnect with backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnect attempts reached for {self.cam_id}")
            self.status = "Disconnected"
            return False
            
        wait_time = min(2 ** self.reconnect_attempts, 30)  # Exponential backoff, max 30s
        logger.info(f"Reconnecting to {self.cam_id} (attempt {self.reconnect_attempts + 1}) - waiting {wait_time}s")
        self.status = "Reconnecting"
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        time.sleep(wait_time)
        self.capture = self._init_capture()
        self.reconnect_attempts += 1
        
        if self.capture and self.capture.isOpened():
            logger.info(f"Successfully reconnected to {self.cam_id}")
            self.reconnect_attempts = 0
            self.status = "Connected"
            return True
        
        return False
    
    def get_frame(self):
        """Get the latest frame with AI processing"""
        with self.frame_lock:
            if self.frame_queue:
                frame = self.frame_queue[-1].copy()  # Get most recent frame
            elif self.last_frame is not None:
                frame = self.last_frame.copy()
            else:
                # Create placeholder frame
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(frame, f"Connecting to {self.cam_id}...", 
                           (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (255, 255, 255), 3)
                cv2.putText(frame, f"Status: {self.status}", 
                           (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                           (255, 255, 255), 2)
                return frame
        
        # Apply AI processing if enabled
        processed_frame = ai_processor.process_frame(frame)
        
        return processed_frame
    
    def stop(self):
        """Stop the camera stream and release resources"""
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        with self.frame_lock:
            self.frame_queue.clear()
            self.last_frame = None
        
        logger.info(f"Stopped camera stream: {self.cam_id}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Camera Streaming Application")
    await initialize_default_cameras()
    yield
    # Shutdown
    logger.info("Shutting down Camera Streaming Application")
    cleanup_streams()

app = FastAPI(
    title="Professional Camera Streaming System",
    description="Real-time camera streaming with AI detection",
    version="3.1.0",
    lifespan=lifespan
)

# Mount static files and templates
templates = Jinja2Templates(directory="templates")

async def initialize_default_cameras():
    """Initialize default cameras from database"""
    global grid_cameras
    
    try:
        response = supabase.table("cameras").select("cam_id").execute()
        existing_ids = [cam['cam_id'] for cam in response.data] if response.data else []
        
        for default_cam in DEFAULT_CAMERAS:
            if default_cam["cam_id"] not in existing_ids:
                try:
                    supabase.table("cameras").insert(default_cam).execute()
                    logger.info(f"Added default camera: {default_cam['cam_id']}")
                except Exception as e:
                    logger.error(f"Error adding camera to database: {str(e)}")
                    continue
            
            if default_cam["cam_id"] not in grid_cameras:
                grid_cameras.append(default_cam["cam_id"])
                start_camera_stream(default_cam["cam_id"], default_cam["cam_url"])
                logger.info(f"Loaded default camera: {default_cam['cam_location']}")
            
    except Exception as e:
        logger.error(f"Error initializing default cameras: {str(e)}")

def cleanup_streams():
    """Clean up all active streams"""
    for cam_id, stream in list(active_streams.items()):
        stream.stop()
    active_streams.clear()
    logger.info("All camera streams cleaned up")

def start_camera_stream(cam_id: str, cam_url: str) -> bool:
    """Start a camera stream"""
    try:
        if cam_id in active_streams:
            active_streams[cam_id].stop()
            del active_streams[cam_id]
            
        stream = CameraStream(cam_id, cam_url)
        if stream.start():
            active_streams[cam_id] = stream
            logger.info(f"Started stream for camera: {cam_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error starting camera stream {cam_id}: {str(e)}")
        return False

def stop_camera_stream(cam_id: str):
    """Stop a camera stream"""
    if cam_id in active_streams:
        active_streams[cam_id].stop()
        del active_streams[cam_id]
        logger.info(f"Stopped stream for camera: {cam_id}")

async def generate_frames(cam_id: str):
    """Generate video frames for streaming"""
    stream = active_streams.get(cam_id)
    if not stream:
        logger.error(f"No active stream for camera: {cam_id}")
        # Create error frame
        error_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(error_frame, f"Camera {cam_id} not found", 
                   (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        _, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(1)
        return
    
    # Wait for initial connection
    await asyncio.sleep(1)
    
    while stream.running:
        try:
            frame = stream.get_frame()
            
            if frame is not None:
                # Encode frame with optimized settings
                encode_params = [
                    cv2.IMWRITE_JPEG_QUALITY, 80,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1
                ]
                _, buffer = cv2.imencode('.jpg', frame, encode_params)
                
                if buffer is not None:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Control streaming rate
            await asyncio.sleep(0.067)  # ~15 FPS
            
        except Exception as e:
            logger.error(f"Error generating frame for {cam_id}: {str(e)}")
            await asyncio.sleep(0.5)

# API Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main dashboard"""
    return templates.TemplateResponse("dashboardtry.html", {"request": request})

@app.get("/api/cameras", response_model=List[Camera])
async def get_cameras():
    """Get all cameras from database"""
    try:
        response = supabase.table("cameras").select("*").execute()
        return response.data if response.data else []
    except Exception as e:
        logger.error(f"Error fetching cameras: {str(e)}")
        raise HTTPException(status_code=500, detail="Error fetching cameras")

@app.post("/api/cameras", response_model=Camera)
async def add_camera(camera: Camera):
    """Add a new camera to database"""
    try:
        response = supabase.table("cameras").insert(camera.dict()).execute()
        if response.data:
            logger.info(f"Camera added: {camera.cam_id}")
            return camera
        else:
            raise HTTPException(status_code=400, detail="Failed to add camera")
    except Exception as e:
        logger.error(f"Error adding camera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/cameras/{cam_id}", response_model=Camera)
async def update_camera(cam_id: str, camera_update: CameraUpdate):
    """Update camera details"""
    try:
        update_data = {k: v for k, v in camera_update.dict().items() if v is not None}
        response = supabase.table("cameras").update(update_data).eq("cam_id", cam_id).execute()
        
        if response.data:
            # Restart stream if URL changed
            if "cam_url" in update_data and cam_id in active_streams:
                stop_camera_stream(cam_id)
                start_camera_stream(cam_id, update_data["cam_url"])
            
            logger.info(f"Camera updated: {cam_id}")
            return response.data[0]
        else:
            raise HTTPException(status_code=404, detail="Camera not found")
    except Exception as e:
        logger.error(f"Error updating camera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/cameras/{cam_id}")
async def delete_camera(cam_id: str):
    """Remove camera from system"""
    try:
        global grid_cameras
        
        if cam_id in active_streams:
            stop_camera_stream(cam_id)
        
        if cam_id in grid_cameras:
            grid_cameras.remove(cam_id)
        
        response = supabase.table("cameras").delete().eq("cam_id", cam_id).execute()
        
        logger.info(f"Camera removed: {cam_id}")
        return {"message": "Camera removed successfully"}
        
    except Exception as e:
        logger.error(f"Error removing camera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stream/{cam_id}")
async def video_stream(cam_id: str):
    """Stream video from specific camera"""
    if cam_id not in active_streams:
        try:
            response = supabase.table("cameras").select("cam_url").eq("cam_id", cam_id).execute()
            if not response.data:
                raise HTTPException(status_code=404, detail="Camera not found")
            
            cam_url = response.data[0]['cam_url']
            
            if not start_camera_stream(cam_id, cam_url):
                raise HTTPException(status_code=500, detail="Failed to start camera stream")
            
            # Wait a moment for stream to initialize
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error starting stream for {cam_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")
    
    return StreamingResponse(
        generate_frames(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.post("/api/ai-toggle")
async def toggle_ai_detection(ai_toggle: AIToggle):
    """Toggle AI detection on/off"""
    global ai_detection_enabled
    
    previous_status = ai_detection_enabled
    ai_detection_enabled = ai_toggle.enabled
    
    status = "enabled" if ai_detection_enabled else "disabled"
    logger.info(f"AI detection {status} (was {'enabled' if previous_status else 'disabled'})")
    
    # Force reload models if enabling AI
    if ai_detection_enabled and not ai_processor.model_loaded:
        logger.info("AI enabled but model not loaded, attempting to reload...")
        ai_processor.load_models()
    
    return {
        "message": f"AI detection {status}", 
        "enabled": ai_detection_enabled,
        "model_loaded": ai_processor.model_loaded,
        "known_faces": len(known_face_names)
    }

@app.get("/api/ai-status")
async def get_ai_status():
    """Get current AI detection status"""
    return {
        "enabled": ai_detection_enabled,
        "yolo_loaded": ai_processor.model_loaded,
        "known_faces": len(known_face_names),
        "model_path": "yolov8n.pt" if ai_processor.model_loaded else "Not loaded"
    }

@app.post("/api/ai/reload")
async def reload_ai_models():
    """Reload AI models and known faces"""
    try:
        logger.info("Reloading AI models...")
        ai_processor.load_models()
        ai_processor.load_known_faces()
        
        return {
            "status": "success",
            "message": "AI models reloaded successfully",
            "yolo_loaded": ai_processor.model_loaded,
            "known_faces": len(known_face_names),
            "face_names": known_face_names
        }
    except Exception as e:
        logger.error(f"Error reloading AI models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/grid-toggle")
async def toggle_grid_camera(grid_toggle: GridToggle):
    """Add/remove cameras from grid view"""
    global grid_cameras
    
    results = []
    
    for cam_id in grid_toggle.camera_ids:
        try:
            if cam_id in grid_cameras:
                # Remove from grid
                grid_cameras.remove(cam_id)
                stop_camera_stream(cam_id)
                results.append(f"Camera {cam_id} removed from grid")
                logger.info(f"Camera removed from grid: {cam_id}")
            else:
                # Add to grid
                response = supabase.table("cameras").select("cam_url").eq("cam_id", cam_id).execute()
                if response.data:
                    grid_cameras.append(cam_id)
                    if start_camera_stream(cam_id, response.data[0]['cam_url']):
                        results.append(f"Camera {cam_id} added to grid")
                        logger.info(f"Camera added to grid: {cam_id}")
                    else:
                        grid_cameras.remove(cam_id)  # Remove if failed to start
                        results.append(f"Failed to start stream for camera {cam_id}")
                else:
                    results.append(f"Camera {cam_id} not found in database")
        except Exception as e:
            logger.error(f"Error toggling camera {cam_id}: {str(e)}")
            results.append(f"Error with camera {cam_id}: {str(e)}")
    
    return {
        "grid_cameras": grid_cameras,
        "results": results
    }

@app.get("/api/grid-status")
async def get_grid_status():
    """Get current grid cameras with their status"""
    camera_status = {}
    
    for cam_id in grid_cameras:
        if cam_id in active_streams:
            stream = active_streams[cam_id]
            camera_status[cam_id] = {
                "status": stream.status,
                "running": stream.running,
                "last_update": stream.last_update_time,
                "reconnect_attempts": stream.reconnect_attempts
            }
        else:
            camera_status[cam_id] = {
                "status": "Not active",
                "running": False,
                "last_update": 0,
                "reconnect_attempts": 0
            }
    
    return {
        "grid_cameras": grid_cameras,
        "camera_status": camera_status,
        "ai_enabled": ai_detection_enabled
    }

@app.get("/api/system-status")
async def get_system_status():
    """Get overall system status"""
    active_count = len(active_streams)
    connected_count = sum(1 for stream in active_streams.values() if stream.status == "Connected")
    
    return {
        "total_cameras": len(grid_cameras),
        "active_streams": active_count,
        "connected_streams": connected_count,
        "ai_detection": {
            "enabled": ai_detection_enabled,
            "model_loaded": ai_processor.model_loaded,
            "known_faces": len(known_face_names)
        },
        "streams_status": {
            cam_id: {
                "status": stream.status,
                "running": stream.running,
                "frame_count": stream.frame_count
            } for cam_id, stream in active_streams.items()
        }
    }

@app.post("/api/camera/{cam_id}/restart")
async def restart_camera_stream(cam_id: str):
    """Restart a specific camera stream"""
    try:
        if cam_id in active_streams:
            # Get camera URL from database
            response = supabase.table("cameras").select("cam_url").eq("cam_id", cam_id).execute()
            if not response.data:
                raise HTTPException(status_code=404, detail="Camera not found")
            
            cam_url = response.data[0]['cam_url']
            
            # Stop existing stream
            stop_camera_stream(cam_id)
            await asyncio.sleep(1)  # Wait a moment
            
            # Start new stream
            if start_camera_stream(cam_id, cam_url):
                return {"message": f"Camera {cam_id} restarted successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to restart camera stream")
        else:
            raise HTTPException(status_code=404, detail="Camera stream not active")
            
    except Exception as e:
        logger.error(f"Error restarting camera {cam_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Set logging level for ffmpeg to reduce HEVC error messages
    os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'  # Suppress most ffmpeg logs
    
    logger.info("Starting Camera Streaming System...")
    logger.info(f"AI Processor initialized - YOLO loaded: {ai_processor.model_loaded}")
    logger.info(f"Known faces loaded: {len(known_face_names)}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5000, 
        reload=False,
        log_level="info"
    )