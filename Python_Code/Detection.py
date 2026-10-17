import cv2
import numpy as np
from collections import deque # For storing up to 64 pairs of centroid coordinates

# Values ​​found in the previous calibration
hsv_min = np.array([0, 153, 128])
hsv_max = np.array([12, 255, 255])

# Buffer for storing the last 64 points of the ball
trace_points = deque(maxlen=64)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# cv2.CAP_DSHOW forces the DirectShow backend on Windows instead of the default.

while True:
    ret, frame = cap.read() #Read the current image from the camera sensor
    # ret (return): (True or False) indicates whether the frame was captured successfully.
    # frame: A numerical matrix containing the pixels of the captured frame.
    if not ret:
        break

    frame = cv2.flip(frame, 1) # The image is inverted so that the X and Y axes align with our perspective.
    height, width, _ = frame.shape # Extract the image resolution
    center_x, center_y = width // 2, height // 2
    # Calculate the optical center of the screen using integer division
    # This point will be used as the setpoint toward which the camera should be aimed.

    # Image processing
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Converts the image from the BGR format to the HSV color space.
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    # Filters the HSV image pixel by pixel, generating a binary image (black and white)
    
    # Filters for removing background noise
    mask = cv2.erode(mask, None, iterations=2) # Reduces the edges of the white regions
    mask = cv2.dilate(mask, None, iterations=2) # Expand the remaining white regions again

    # Find contours
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # mask.copy(): Creates a copy of the original mask
    # cv2.RETR_EXTERNAL: Extract only the outermost contours, ignoring any contours or voids contained within them.
    # cv2.CHAIN_APPROX_SIMPLE: Compresses the horizontal, vertical, and diagonal segments, leaving only their endpoints.
    centroid = None

    if len(cnts) > 0:
        # Isolate the largest contour based on enclosed surface area
        c = max(cnts, key=cv2.contourArea) #Select the contour with the largest area, ignoring minor distractions.
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        # Calculate the minimum enclosing circle of the ball and return its approximate center and its radius in pixels.
        M = cv2.moments(c) # Calculate the spatial moments of the silhouette
        # M["m00"]: Total number of white pixels
        # M["m10"]: The sum of all the X positions of the white pixels.
        # M["m01"]: The sum of all the Y positions of the white pixels.

        if M["m00"] > 0 and radius > 10: # Verify that the object is physically meaningful.
            # Centroid (xc, yc) by spatial moments
            xc = int(M["m10"] / M["m00"])
            yc = int(M["m01"] / M["m00"])
            centroid = (xc, yc)

            # Draw outline and centroid
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            # Draw the yellow circle, with the value of the center and the radius
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
            # Draw the red centroid of the circle

            # Pixel error relative to the camera center
            error_x = center_x - xc
            error_y = center_y - yc

            # Display telemetry on screen
            cv2.putText(frame, f"Error X: {error_x} px | Error Y: {error_y} px", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw the historical trace (red line)
    trace_points.appendleft(centroid) # Insert the newly calculated centroid at position 0 of the queue
    for i in range(1, len(trace_points)):
        if trace_points[i - 1] is None or trace_points[i] is None:
            continue
        thickness = int(np.sqrt(64 / float(i + 1)) * 1.5) # The thickness formula tapers the line as the point moves further back along the tail.
        cv2.line(frame, trace_points[i - 1], trace_points[i], (0, 0, 255), max(1, thickness))
        # This creates a dynamic visual effect where the head of the trail is thick and the tail becomes progressively thinner.

    #  Draw the reference point (SetPoint) cross
    cv2.line(frame, (center_x - 15, center_y), (center_x + 15, center_y), (255, 255, 255), 1)
    cv2.line(frame, (center_x, center_y - 15), (center_x, center_y + 15), (255, 255, 255), 1)
    #  Draw two intersecting lines, 30 pixels in length, right at the center of the video
    cv2.imshow("Detect & Follow - PC Prototype", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()

