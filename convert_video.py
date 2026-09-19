import subprocess
import imageio_ffmpeg

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

subprocess.run([
    ffmpeg,
    "-i", "runs/detect/predict2/traffic.avi",
    "-c:v", "libx264",
    "-c:a", "aac",
    "output/videos/helmet_detection.mp4"
])

print("Conversion completed!")