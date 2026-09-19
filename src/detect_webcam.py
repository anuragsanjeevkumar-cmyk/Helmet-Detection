from ultralytics import YOLO

# Load our trained model
model = YOLO("model/best.pt")

# Start webcam detection
model.predict(
    source=0,
    conf=0.25,
    show=True
)