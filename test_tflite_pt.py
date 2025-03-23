import cv2
import time
import numpy as np
import tensorflow.lite as tflite
from ultralytics import YOLO
import ncnn
from cv2.dnn import NMSBoxes as nms

print("Import done")

# Load YOLO model
yolo_model = YOLO("/home/suja/Desktop/picar/best_ncnn_model", task='detect')


# Load TFLite model
interpreter = tflite.Interpreter(model_path='quant_model_cnn_test.tflite', num_threads=4)
interpreter.allocate_tensors()

# Get input/output tensor indexes
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Quantization:", input_details[0]['quantization'])
print("Output Quantization:", output_details[0]['quantization'])
print("\n\nLoad success")

label = {0: '30km/h', 1: '40km/h', 2: '60km/h', 3: '80km/h', 4: 'pedestrian', 5: 'school zone'}

prev_frame_time = 0
new_frame_time = 0

def preprocess_for_cnn(cropped_img):
    resized = cv2.resize(cropped_img, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    normalized = resized.astype(np.float32) / 255.0
    input_scale, input_zero_point = input_details[0]["quantization"]
    image = normalized / input_scale + input_zero_point
    return np.clip(image, -128, 127).astype(np.int8)[np.newaxis, ...]

def dequantize_output(output_data):
    output_scale, output_zero_point = output_details[0]["quantization"]
    return (output_data.astype(np.float32) - output_zero_point) * output_scale
def preprocess_for_ncnn(img):
    """Preprocess image for NCNN input"""
    # Resize and convert to correct format
    resized = cv2.resize(img, (640, 480))
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Convert to float32 and normalize
    normalized = img_rgb.astype(np.float32) / 255.0
    normalized = np.transpose(normalized, (2, 0, 1))
    normalized = np.expand_dims(normalized, axis=0)
    return normalized
def process_onnx_output(output, conf_threshold=0.95,iou_threshold=0.2):
    """Process ONNX output from shape (1, 10, 8400) to (n, 6)"""
    # Reshape to (batch, elements, anchors)
    # batch, elements, anchors = output.shape
    output = output.T
    print(f"Output shape after transpose: {output.shape}")
    # Separate box coordinates and class predictions
    boxes = np.array(output[:, :4] )
    print(f"{boxes = }")
    bboxes = []
    for box in boxes:
         x_center, y_center, width, height = box
         x1 = int((x_center - width / 2))
         y1 = int((y_center - height / 2))
         x2 = int((x_center + width / 2) )
         y2 = int((y_center + height / 2))
         bboxes.append([x1, y1, x2, y2])

    scores = np.max(output[:, 4:], axis=1)
    print(f"{scores = }")
    print(max(scores))
    classes = np.argmax(output[:, 4:], axis=1)
    print(f"{classes = }")
    indices = cv2.dnn.NMSBoxes(bboxes, scores, conf_threshold, iou_threshold,top_k=1)
    print("Indices:", indices)
    print("Indices shape:", indices.shape)
    if indices:
        detections = bboxes(indices[0])
    
    return detections
yolo_times = []
cnn_times = []
frames = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 5)    
time.sleep(2)

while frames < 20:
    frames += 1
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps}")

    # YOLO Detection
    yolo_start = time.time()
    ncnn_input = preprocess_for_ncnn(frame)
    ncnn_input = np.ascontiguousarray(ncnn_input)
    ncnn_input = ncnn.Mat.from_pixels(ncnn_input, ncnn.Mat.PixelType.PIXEL_BGR, ncnn_input.shape[1], ncnn_input.shape[0])
    with ncnn_net.create_extractor() as ex:
    # Convert numpy array to NCNN Mat
        mat_in = ncnn.Mat(ncnn_input)
        # Input with name matching your NCNN model
        ex.input("in0", mat_in)
        # Extract output with name matching your NCNN model
        ret, ncnn_output = ex.extract("out0")
    # Convert output to numpy array for processing
    ncnn_output = np.array(ncnn_output)
    print("NCNN output shape:", ncnn_output.shape)
    # ncnn_output = np.expand_dims(ncnn_output, axis=0)
    # print("NCNN output shape after expd:", ncnn_output.shape)
    detections = process_onnx_output(ncnn_output)
    yolo_end = time.time()
    yolo_times.append(yolo_end - yolo_start)
    print("Detection done")

    print("Detections:", detections)

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection[:4])
        cropped_img = frame[y1:y2, x1:x2]
        if cropped_img.size == 0:
            print("Skipping empty crop")
            continue

        # CNN Classification
        cnn_start = time.time()
        preprocessed_img = preprocess_for_cnn(cropped_img)
        interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        prediction = dequantize_output(prediction)
        cnn_end = time.time()
        cnn_times.append(cnn_end - cnn_start)

        predicted_class = np.argmax(prediction)
        # print(label[predicted_class])
        cv2.putText(frame, label[predicted_class], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""---- Printing stats ----"""
print("Average YOLO Detection Time: ", np.mean(yolo_times))
print("Average CNN Detection Time: ", np.mean(cnn_times))
print("Total Frames: ", frames)

cap.release()
cv2.destroyAllWindows()