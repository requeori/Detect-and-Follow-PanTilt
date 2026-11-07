import cv2
import time #CPU's internal clock
import numpy as np #Allows for performing vector mathematical operations on the accumulated samples

CAM_INDEX = 1
# CAM_INDEX = 0 : Emeet S800 camera
# CAM_INDEX = 1 : Aukey PC-LM1E camera

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
# cv2.CAP_DSHOW forces the DirectShow backend on Windows instead of the default.

# Resolution settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 30 FPS is set in the driver.
cap.set(cv2.CAP_PROP_FPS, 30)

frame_times = []
# Create a list "frame_times" in which the time taken for each frame capture and processing operation will be stored.
num_frames = 150
# 150 frames will be analyzed. 

print("Measuring vision pipeline performance.....")

for i in range(num_frames): # Creates a loop from 0 to 150 frames
    t_start = time.perf_counter() # Start the timer just before requesting the image from the camera.
    
    ret, frame = cap.read() #Read the current image from the camera sensor
    # ret (return): (True or False) indicates whether the frame was captured successfully.
    # frame: A numerical matrix containing the pixels of the captured frame.
    if not ret:
        print("Error accessing the camera")
        break

    # Vision processing load
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Converts the image from the BGR format to the HSV color space.
    mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
    # Filters the HSV image pixel by pixel, generating a binary image (black and white).
    # The limits "np.array([85, 255, 255])" and "np.array([35, 80, 80])" were defined this way only  for the test.
    mask = cv2.erode(mask, None, iterations=2) # Reduces the edges of the white regions
    mask = cv2.dilate(mask, None, iterations=2) # Expand the remaining white regions again
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Analyze the black and white mask to locate the bounding edges of all the white areas remaining after filtering.
    # cv2.RETR_EXTERNAL: Extract only the outer contours of the objects, ignoring internal voids within the detected area.
    # cv2.CHAIN_APPROX_SIMPLE: Compresses the points along the perimeter, keeping only the essential vertices
    t_end = time.perf_counter() # Stops the timer once all frame processing is complete.
    frame_times.append((t_end - t_start) * 1000)
    # Calculate the time difference, multiply it by 1000 to convert to milliseconds, and store the result in the list.

cap.release() #Closes the connection to the camera

# Calculation of metrics
frame_times = np.array(frame_times)
real_fps = 1000.0 / np.mean(frame_times) # Converts the average time between frames into frames per second
dt_average = np.mean(frame_times) # Calculates the average time
dt_jitter = np.std(frame_times) # Measures the dispersion or variability of the time between frames.

print("\n--- BENCHMARK RESULTS ---")
print(f"Actual FPS under load: {real_fps:.2f} FPS")
print(f"Average latency per frame: {dt_average:.2f} ms")
print(f"Jitter (Variation/Standard Deviation): {dt_jitter:.2f} ms")

