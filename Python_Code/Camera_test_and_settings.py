import cv2

# There are currently 2 webcams available, and tests are to be conducted to determine which one to use.
# Latency tests are planned, but first, certain driver parameters must be disabled.
# The variable CAM_INDEX is used to specify which camera to test, where:
CAM_INDEX = 0
# CAM_INDEX = 0 : Emeet S800 camera
# CAM_INDEX = 1 : Aukey PC-LM1E camera

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
# cv2.CAP_DSHOW forces the DirectShow backend on Windows instead of the default.

cap.set(cv2.CAP_PROP_SETTINGS, 1)  # To open windows settings panel.
# When the script is run, a floating Windows window will open.
# In this window, you can disable auto-exposure, white balance, or autofocus.
# It is necessary to disable these settings so that they do not negatively affect image detection.

# Resolution settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read() #Read the current image from the camera sensor
    # ret (return): (True or False) indicates whether the frame was captured successfully.
    # frame: A numerical matrix containing the pixels of the captured frame.
    if not ret:
        print("Error accessing the camera")
        break
    cv2.imshow("Camera test - Manual settings", frame)
    # Creates a window with the specified title to display the processed frame in real time.
    if cv2.waitKey(1) & 0xFF == ord('q'): # Process window events and detect exit ('q' key)
        break
cap.release() #Closes the connection to the camera
cv2.destroyAllWindows() #Closes all graphical windows that have been created
