import cv2
import time
import datetime
import threading

# Initialize cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

# Detection and recording parameters
SECONDS_TO_RECORD_AFTER_DETECTION = 10
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

def process_camera(cam_index):
    cam = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cam.isOpened():
        print(f"Camera {cam_index} could not be opened.")
        return

    detection = False
    detection_stopped_time = None
    timer_started = False
    out = None

    frame_size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Validate frame size
    if frame_size[0] == 0 or frame_size[1] == 0:
        print(f"Camera {cam_index} has invalid frame size.")
        cam.release()
        return

    while True:
        ret, frame = cam.read()
        if not ret:
            print(f"Camera {cam_index} disconnected.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces and bodies
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

        detected = len(faces) > 0 or len(bodies) > 0

        current_time = time.time()

        # Handle detection logic
        if detected:
            if not detection:
                # Start recording if detection starts
                detection = True
                current_time = datetime.datetime.now()
                out = cv2.VideoWriter(f"Camera{cam_index}_Security_{current_time.strftime('%Y%m%d_%H%M%S')}.mp4", fourcc, 20, frame_size)
                print(f"Started Recording on Camera {cam_index}")
                timer_started = False  # Reset timer if detection starts
            else:
                # Reset the timer if detection is still ongoing
                timer_started = False
        elif detection:
            # Stop recording if no detection and the timer exceeds SECONDS_TO_RECORD_AFTER_DETECTION
            if not timer_started:
                timer_started = True
                detection_stopped_time = current_time
            elif current_time - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                if out:
                    out.release()
                    print(f"Stopped Recording on Camera {cam_index}")

        if detection and out:
            out.write(frame)

        # Draw rectangles around detected faces and bodies
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        cv2.imshow(f"Camera {cam_index}", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    if out:
        out.release()
    cam.release()
    cv2.destroyAllWindows()
    print(f"Camera {cam_index} thread ended.")

if __name__ == "__main__":
    print("Select cameras to open:")
    print("0: Default Camera")
    print("1: External Camera (if available)")
    print("Enter camera indices separated by commas (e.g., 0,1):")

    camera_indices = input().split(",")
    threads = []

    for cam_index in camera_indices:
        try:
            cam_index = int(cam_index.strip())
            thread = threading.Thread(target=process_camera, args=(cam_index,))
            threads.append(thread)
            thread.start()
        except ValueError:
            print(f"Invalid input: {cam_index}")

    for thread in threads:
        thread.join()
