from ultralytics import YOLO

# Load our trained helmet detection model
model = YOLO("model/best.pt")

# Run detection on an image
results = model.predict(
    source="input/images/test.jpg",
    conf=0.25,
    save=True
)

print("Detection completed!")