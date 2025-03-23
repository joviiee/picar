import multiprocessing as mp
import threading
import cv2
import tensorflow.lite as tflite
import numpy as np
from ultralytics import YOLO
import time
import queue

def initialize_camera():
    """Initialize either PiCamera2 or USB camera based on USE_PICAM setting"""
    if USE_PICAM:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Wait for camera to initialize
        return picam2
    else:
        cap = cv2.VideoCapture(0)
        time.sleep(2)
        if not cap.isOpened():
            raise RuntimeError("Failed to open USB camera")
        return cap
    
def capture_frame(camera):
    """Capture a frame from either PiCamera2 or USB camera"""
    if USE_PICAM:
        frame = camera.capture_array()
        detection_queue.put(frame)
    else:
        frame =  camera.read()
        detection_queue.put(frame)

def release_camera(camera):
    """Properly close the camera"""
    if USE_PICAM:
        camera.close()
    else:
        camera.release()

def preprocess_for_cnn(cropped_img):
    resized = cv2.resize(cropped_img, (input_details[0]['shape'][1], input_details[0]['shape'][2])) 
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0).astype(np.int8)

def detect(input_queue, output_queue,stop_event,yolo_ready):
    global yolo_model
    
    yolo_model = YOLO("best_ncnn_model=",task='detect')
    _ = yolo_model.predict(np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8), conf=0.5,verbose=False) 
    yolo_ready.set()
    print("yolo loaded")
    while not stop_event.is_set():
        try:
            frame = input_queue.get()  # Try getting the latest frame
        except queue.Empty:  
            continue  # If queue is empty, just skip this iteration
        # Run NCNN object detection
        results = yolo_model.predict(frame, conf=0.5,verbose=False)  
        detections = results[0].boxes.data.cpu().numpy()
        print("detection done")
        print(len(detections))
        output_queue.put([frame, detections])

def classify(input_queue,stop_event,cnn_ready):
    global interpreter
    global input_details
    global output_details
    
    interpreter = tflite.Interpreter(
        model_path="/home/suja/Desktop/picar/quant_model_cnn_test.tflite"
    )
    interpreter.allocate_tensors()

    # Get input/output tensor indexes
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    cnn_ready.set()
    print("cnn loaded")
    while not stop_event.is_set():
        try:
            frame, detections = input_queue.get()  # Try getting the latest frame
        except queue.Empty:  
            continue  # If queue is empty, just skip this iteration
        # Run CNN classification on detected objects
        for detection in detections:
            x1, y1, x2, y2, conf, cls = map(int, detection[:6])  

            cropped_img = frame[y1:y2, x1:x2]

            preprocessed_img = preprocess_for_cnn(cropped_img)
            
            interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            predicted_class = np.argmax(prediction)
            
            print("classification done")

            print(label[predicted_class])

def capture_frames(input_queue,stop_event):
    cap = cv2.VideoCapture(0)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            try:
                input_queue.get_nowait()
                input_queue.put(frame)
            except queue.Empty:
                input_queue.put(frame) 
        print("image ...")
        time.sleep(0.1)

if __name__ == "__main__":
    USE_PICAM = False

    # yolo_model = YOLO("best_ncnn_model=",task='detect') 
    # print("yolo loaded")
    # interpreter = tflite.Interpreter(
    #     model_path="/home/suja/Desktop/picar/quant_model_cnn_test.tflite"
    # )
    # print("cnn loaded")
    # interpreter.allocate_tensors()

    # # Get input/output tensor indexes
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    label={0:'30km/h',1:'40km/h',2:'60km/h',3:'80km/h',4:'pedestrain',5:'school zone'}
    
    stop_event = mp.Event()
    yolo_ready = mp.Event()
    cnn_ready = mp.Event()
    
    detection_queue = mp.Queue(maxsize=1)
    classification_queue = mp.Queue(maxsize=1)
    print("Q")
    p1 = mp.Process(target=detect, args=(detection_queue, classification_queue,stop_event,yolo_ready))
    p2 = mp.Process(target=classify, args=(classification_queue,stop_event,cnn_ready))

    p1.start()
    p2.start()
    
    yolo_ready.wait()
    cnn_ready.wait()
    
    time.sleep(5)
    
    # Start the camera capture in a separate thread
    capture_thread = threading.Thread(target=capture_frames, args=(detection_queue,stop_event),daemon=True).start()
    
    try:
        while True:
            pass  # Keep main process running
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt detected. Cleaning up...")

        # Signal all processes to stop
        stop_event.set()

        # Clear queues to avoid hanging
        while not detection_queue.empty():
            detection_queue.get()
        while not classification_queue.empty():
            classification_queue.get()

        # Wait for processes to terminate
        p1.join()
        p2.join()

        # Close queues (optional, but good practice)
        detection_queue.close()
        classification_queue.close()
