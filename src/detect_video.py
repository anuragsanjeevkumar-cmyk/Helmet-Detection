from ultralytics import YOLO

# Load our trained model
model = YOLO("model/best.pt")

# Run detection on video
results = model.predict(
    source="input/videos/traffic.mp4",
    conf=0.25,
    save=True
)

print("Video detection completed!")