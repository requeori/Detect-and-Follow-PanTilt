import cv2
import numpy as np

def nothing(x): # An empty function is created, which will be used as the fifth parameter in the sliders
# This empty function is necessary since nothing will automatically move the sliders.
    pass

cv2.namedWindow("HSV Calibrator") # Create a window with trackbars.
cv2.createTrackbar("H Min", "HSV Calibrator", 35, 179, nothing)  #Slider for the H min value.
cv2.createTrackbar("H Max", "HSV Calibrator", 85, 179, nothing)  #Slider for the H max value.
cv2.createTrackbar("S Min", "HSV Calibrator", 80, 255, nothing)  #Slider for the S min value.
cv2.createTrackbar("S Max", "HSV Calibrator", 255, 255, nothing) #Slider for the S max value.
cv2.createTrackbar("V Min", "HSV Calibrator", 80, 255, nothing)  #Slider for the V min value.
cv2.createTrackbar("V Max", "HSV Calibrator", 255, 255, nothing) #Slider for the V max value.

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# cv2.CAP_DSHOW forces the DirectShow backend on Windows instead of the default.

while True:
    ret, frame = cap.read() #Read the current image from the camera sensor
    # ret (return): (True or False) indicates whether the frame was captured successfully.
    # frame: A numerical matrix containing the pixels of the captured frame.
    if not ret:
        break

    frame = cv2.flip(frame, 1) # The image is inverted so that the X and Y axes align with our perspective.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #Converts the image from the BGR format to the HSV color space.

    # Read values from the sliders
    h_min = cv2.getTrackbarPos("H Min", "HSV Calibrator")
    h_max = cv2.getTrackbarPos("H Max", "HSV Calibrator")
    s_min = cv2.getTrackbarPos("S Min", "HSV Calibrator")
    s_max = cv2.getTrackbarPos("S Max", "HSV Calibrator")
    v_min = cv2.getTrackbarPos("V Min", "HSV Calibrator")
    v_max = cv2.getTrackbarPos("V Max", "HSV Calibrator")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    # Pack the positions into two vectors that define the lower and upper bounds of the mask.

    # Generate binary mask
    mask = cv2.inRange(hsv, lower, upper)
    # Filters the HSV image pixel by pixel, generating a binary image (black and white)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    # Compare the mask image with the original, allowing you to see the selected color against a black background.
    
    cv2.imshow("Original Camera", frame) # Show the original image
    cv2.imshow("Binary Mask (HSV)", mask) # Show the Binary Mask
    cv2.imshow("Segmented Object", result) # Displays the segmented image, with only the selected color.

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Saved ranges: MIN=[{h_min}, {s_min}, {v_min}]  MAX=[{h_max}, {s_max}, {v_max}]")
        # When we press the "q" key, it returns the values ​​we have set.
        break

cap.release()
cv2.destroyAllWindows()

