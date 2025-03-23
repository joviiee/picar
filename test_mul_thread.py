import cv2
import time
import numpy as np
import tensorflow.lite as tflite
from ultralytics import YOLO
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignDetectionSystem:
    def __init__(self):
        self.frame_queue = Queue(maxsize=10)
        self.detection_queue = Queue(maxsize=5)
        self.classification_queue = Queue(maxsize=5)
        self.running = True
        self.fps = 0
        self.label = {0:'30km/h', 1:'40km/h', 2:'60km/h', 3:'80km/h',
                     4:'pedestrian', 5:'school zone'}
        
        # Initialize models
        self.yolo_model = YOLO("best.pt")
        self.interpreter = tflite.Interpreter(model_path='quant_model_cnn_test.tflite')
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def capture_frames(self):
        """Camera capture function"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            # logger.info("Captured frame")
            logger.info(f"frame queue size: {self.frame_queue.qsize()}")
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
        cap.release()

    def detect_signs(self):
        """YOLO detection function"""
        logger.info("Starting YOLO detection...")
        while self.running:
            logger.info(f"frame queue size: {self.frame_queue.qsize()}")
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                results = self.yolo_model.predict(frame, conf=0.5, verbose=False)
                if results and len(results) > 0:
                    logger.info("Detected sign into queue")
                    self.detection_queue.put((frame, results[0].boxes.data.cpu().numpy()))

    def classify_signs(self):
        """TFLite classification function"""
        while self.running:
            logger.info(f"detection queue size: {self.detection_queue.qsize()}")
            if not self.detection_queue.empty():
                frame, detections = self.detection_queue.get()
                for detection in detections:
                    x1, y1, x2, y2, conf, cls = map(int, detection[:6])
                    cropped_img = frame[y1:y2, x1:x2]
                    if cropped_img.size > 0:
                        try:
                            preprocessed = self.preprocess_image(cropped_img)
                            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
                            self.interpreter.invoke()
                            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
                            prediction = self.dequantize_output(prediction)
                            logger.info(f"Classification prediction: {prediction}")
                            self.classification_queue.put((frame, (x1, y1, x2, y2), prediction))
                        except Exception as e:
                            logger.error(f"Classification failed: {e}")

    def preprocess_image(self, img):
        resized = cv2.resize(img, (self.input_details[0]['shape'][1], 
                                 self.input_details[0]['shape'][2]))
        normalized = resized / 255.0
        input_scale, input_zero_point = self.input_details[0]["quantization"]
        image = normalized / input_scale + input_zero_point
        return np.clip(image, -128, 127).astype(np.int8)[np.newaxis, ...]

    def dequantize_output(self, output_data):
        scale, zero_point = self.output_details[0]["quantization"]
        return (output_data.astype(np.float32) - zero_point) * scale

    def run(self):
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit tasks
            futures = [
                executor.submit(self.capture_frames),
                executor.submit(self.detect_signs),
                executor.submit(self.classify_signs)
            ]
            
            frame_count = 0
            prev_time = time.time()
            
            try:
                while self.running:
                    logger.info("Main loop")
                    logger.info(f"classification queue size: {self.classification_queue.qsize()}")
                    if not self.classification_queue.empty():
                        frame, (x1, y1, x2, y2), prediction = self.classification_queue.get()
                        predicted_class = np.argmax(prediction)
                        
                        # Draw bounding box and label
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label_text = f"{self.label[predicted_class]} {prediction[0][predicted_class]:.2f}"
                        logger.info(f"Label: {label_text}")
                        # cv2.putText(frame, label_text, (x1, y1 - 10),
                        #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Update FPS
                        frame_count += 1
                        current_time = time.time()
                        if current_time - prev_time >= 1.0:
                            self.fps = frame_count / (current_time - prev_time)
                            frame_count = 0
                            prev_time = current_time
                            logger.info(f"FPS: {self.fps:.2f}")
                        
                        # Display FPS on frame
                        # cv2.putText(frame, f'FPS: {self.fps:.2f}', (10, 30),
                        #           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Show frame
                        # cv2.imshow("Traffic Speed Sign Detection", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
            except KeyboardInterrupt:
                logger.info("Stopping application...")
            finally:
                self.running = False
                cv2.destroyAllWindows()
                
                # Wait for all tasks to complete
                for future in futures:
                    future.result()

def main():
    system = SignDetectionSystem()
    system.run()

if __name__ == "__main__":
    main()