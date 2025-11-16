from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import asyncio
from typing import List, Dict, Optional
import json
import os
from datetime import datetime

from models.drowning_detector import DrowningDetector
from models.pose_analyzer import PoseAnalyzer
from services.alert_service import AlertService
from services.stream_processor import StreamProcessor

app = FastAPI(title="BeachSafe AI Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
drowning_detector = DrowningDetector()
pose_analyzer = PoseAnalyzer()
alert_service = AlertService()
stream_processor = StreamProcessor(drowning_detector, pose_analyzer, alert_service)

# Global state for streams
active_streams: Dict[str, dict] = {}
alert_history: List[dict] = []

@app.get("/")
async def root():
    return {"message": "BeachSafe AI Backend", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/streams/start")
async def start_stream(stream_config: dict):
    """Start processing a video stream"""
    stream_id = stream_config.get("stream_id", f"stream_{len(active_streams)}")
    stream_url = stream_config.get("url")

    if not stream_url:
        raise HTTPException(status_code=400, detail="Stream URL required")

    try:
        await stream_processor.start_stream(stream_id, stream_url)
        active_streams[stream_id] = {
            "url": stream_url,
            "status": "active",
            "started_at": datetime.now().isoformat()
        }
        return {"message": f"Stream {stream_id} started", "stream_id": stream_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")

@app.post("/api/streams/stop/{stream_id}")
async def stop_stream(stream_id: str):
    """Stop processing a video stream"""
    try:
        await stream_processor.stop_stream(stream_id)
        if stream_id in active_streams:
            active_streams[stream_id]["status"] = "stopped"
            active_streams[stream_id]["stopped_at"] = datetime.now().isoformat()
        return {"message": f"Stream {stream_id} stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop stream: {str(e)}")

@app.get("/api/streams/status")
async def get_streams_status():
    """Get status of all active streams"""
    return {
        "active_streams": active_streams,
        "stream_count": len(active_streams)
    }

@app.get("/api/alerts/history")
async def get_alert_history(limit: int = 50):
    """Get recent alert history"""
    return {"alerts": alert_history[-limit:]}

@app.post("/api/alerts/send")
async def send_manual_alert(alert_data: dict):
    """Send a manual alert"""
    try:
        result = await alert_service.send_alert(
            phone=alert_data.get("phone"),
            email=alert_data.get("email"),
            location=alert_data.get("location", "Whale Beach"),
            likelihood=alert_data.get("likelihood", "High"),
            stream_id=alert_data.get("stream_id")
        )
        alert_record = {
            "timestamp": datetime.now().isoformat(),
            "type": "manual",
            "data": alert_data,
            "result": result
        }
        alert_history.append(alert_record)
        return {"message": "Alert sent", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send alert: {str(e)}")

@app.post("/api/training/save")
async def save_training_data(file: UploadFile = File(...), label: str = "drowning"):
    """Save training data"""
    try:
        # Create dataset directory if it doesn't exist
        os.makedirs(f"dataset/{label}", exist_ok=True)

        # Save the uploaded file
        file_path = f"dataset/{label}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return {"message": f"Training data saved to {label} dataset", "path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save training data: {str(e)}")

@app.websocket("/ws/stream/{stream_id}")
async def stream_websocket(websocket: WebSocket, stream_id: str):
    """WebSocket endpoint for real-time stream data"""
    await websocket.accept()

    try:
        while True:
            # Get latest detection data for this stream
            detection_data = stream_processor.get_stream_data(stream_id)
            if detection_data:
                await websocket.send_json(detection_data)
            await asyncio.sleep(0.1)  # 10 FPS updates
    except Exception as e:
        print(f"WebSocket error for stream {stream_id}: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)