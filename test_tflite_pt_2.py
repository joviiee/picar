import cv2
import time 
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
from ultralytics import YOLO
print("import done")

yolo_model = YOLO("best_float32.tflite",task='detect') 
# Load TFLite model
interpreter = tf.lite.Interpreter(
    model_path="/home/suja/Desktop/picar/quant_model_cnn_test.tflite"
)
interpreter.allocate_tensors()

# Get input/output tensor indexes
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n\nload success")

label={0:'30km/h',1:'40km/h',2:'60km/h',3:'80km/h',4:'pedestrain',5:'school zone'}

prev_frame_time = 0
new_frame_time = 0

def preprocess_for_cnn(cropped_img):
    resized = cv2.resize(cropped_img, (input_details[0]['shape'][1], input_details[0]['shape'][2])) 
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=0).astype(np.int8)

USE_PICAM = False  # Set to False to use USB camera

def initialize_camera():
    """Initialize either PiCamera2 or USB camera based on USE_PICAM setting"""
    if USE_PICAM:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (1280, 720)})
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
        return True, frame
    else:
        return camera.read()

def release_camera(camera):
    """Properly close the camera"""
    if USE_PICAM:
        camera.close()
    else:
        camera.release()


cap = initialize_camera()  

while True:
    ret, frame = capture_frame(cap)
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    print("fps = ",fps)
    
    # Convert fps to string to display
    fps_text = f'FPS: {fps}'
    # print(frame)

    results = yolo_model.predict(frame, conf=0.5,verbose=False) 
    # print(f"results = {results[0].boxes.data.cpu().numpy()}")
    detections = results[0].boxes.data.cpu().numpy()  
    # print("detection done ")
    # print(detections)
    for detection in detections:
        x1, y1, x2, y2, conf, cls = map(int, detection[:6])  
        class_id = int(cls)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cropped_img = frame[y1:y2, x1:x2]
        # print(cropped_img)
        preprocessed_img = preprocess_for_cnn(cropped_img)
        
        # Run inference without batch processing (batch size = 1)
        interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        # print(prediction)
        predicted_class = np.argmax(prediction)

        prediction_label = f"class: {predicted_class}"
        # print(prediction_label)
        print(label[predicted_class])
        cv2.putText(frame, label[predicted_class], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # cv2.imshow("Traffic Speed Sign Detection", frame)
    print("image displayed")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

release_camera(cap)
cv2.destroyAllWindows()
