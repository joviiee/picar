import cv2
import numpy as np
import ncnn
import time
import logging
import sys

np.set_printoptions(threshold=sys.maxsize, suppress=True, precision=6)

log_filename = f"ncnn_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # This will also print to console
    ]
)
logger = logging.getLogger(__name__)

def preprocess_for_ncnn(img):
    resized = cv2.resize(img, (640, 480))
    img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = img_rgb.astype(np.float32) / 255.0
    normalized = np.transpose(normalized, (2, 0, 1))
    normalized = np.expand_dims(normalized, axis=0)
    return normalized


# def process_onnx_output(output, conf_threshold=0.8):
#     """Process ONNX output from shape (1, 10, 8400) to (n, 6)"""
#     # Reshape to (batch, elements, anchors)
#     # batch, elements, anchors = output.shape
    
#     # Separate box coordinates and class predictions
#     boxes = output[:, :4, :]  # First 4 elements are bbox coords
#     class_scores = output[:, 4:, :]  # Remaining elements are class scores
    
#     # Reshape boxes to (batch, anchors, 4)
#     boxes = np.transpose(boxes, (0, 2, 1))
    
#     # Reshape scores to (batch, anchors, num_classes)
#     class_scores = np.transpose(class_scores, (0, 2, 1))
    
#     # Get confidence scores and class ids
#     confidences = np.max(class_scores, axis=2)
#     class_ids = np.argmax(class_scores, axis=2)
    
#     # Filter by confidence threshold
#     mask = confidences[0] > conf_threshold
#     filtered_boxes = boxes[0][mask]
#     filtered_scores = confidences[0][mask]
#     filtered_class_ids = class_ids[0][mask]
    
#     # Stack results to match YOLO format (n, 6)
#     detections = np.column_stack((
#         filtered_boxes,
#         filtered_scores,
#         filtered_class_ids
#     ))
    
#     return detections

def non_max_suppression(boxes, scores, iou_threshold):
    # Sort boxes by scores in descending order
    sorted_indices = scores.argsort()[::-1]
    keep_boxes = []
    
    while sorted_indices.size > 0:
        # Pick the box with highest score
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        
        # Calculate IoU of the picked box with rest of the boxes
        ious = compute_iou(boxes[box_id], boxes[sorted_indices[1:]])
        
        # Remove boxes with IoU over threshold
        keep_indices = np.where(ious < iou_threshold)[0]
        
        # Update indices
        sorted_indices = sorted_indices[keep_indices + 1]
    
    return keep_boxes

def compute_iou(box, boxes):
    # box: single box [x1, y1, x2, y2]
    # boxes: multiple boxes [N, 4]
    
    # Calculate intersection
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate union
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    
    return intersection / union

def process_raw_output(output, conf_threshold=0.85, iou_threshold=0.2):
    """Process raw YOLO output with shape (10, 8400)"""
    # Get scores and class predictions
    boxes = output[:4, :].T  # First 4 rows are bbox coords (x,y,w,h), transpose to (8400, 4)
    scores = output[4:, :].T  # Remaining rows are class scores, transpose to (8400, 6)
    
    # Get class scores and indices
    class_scores = np.max(scores, axis=1)  # Get max score for each detection
    class_ids = np.argmax(scores, axis=1)  # Get class id with max score
    
    # Filter by confidence threshold
    mask = class_scores > conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = class_scores[mask]
    filtered_class_ids = class_ids[mask]
    
    # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)
    x_center = filtered_boxes[:, 0]
    y_center = filtered_boxes[:, 1]
    width = filtered_boxes[:, 2]
    height = filtered_boxes[:, 3]
    
    x1 = x_center - width/2
    y1 = y_center - height/2
    x2 = x_center + width/2
    y2 = y_center + height/2
    
    filtered_boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # Apply NMS for each class
    keep_boxes = []
    unique_classes = np.unique(filtered_class_ids)
    
    for class_id in unique_classes:
        class_mask = filtered_class_ids == class_id
        class_boxes = filtered_boxes[class_mask]
        class_scores = filtered_scores[class_mask]
        
        # Apply NMS
        keep_indices = non_max_suppression(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(np.where(class_mask)[0][keep_indices])
    
    # Get final detections
    keep_boxes = np.array(keep_boxes)
    final_boxes = filtered_boxes[keep_boxes]
    final_scores = filtered_scores[keep_boxes]
    final_class_ids = filtered_class_ids[keep_boxes]
    
    # Stack results into single array [x1,y1,x2,y2,conf,class_id]
    detections = np.column_stack((
        final_boxes,
        final_scores,
        final_class_ids
    ))
    
    return detections

# Update the main loop to use the new processing function

# def postprocess(predictions, confidence_threshold=0.5, iou_threshold=0.5):
#     # Assuming predictions contains boxes and scores
#     boxes = predictions['boxes']  # Shape: [N, 4] with format [x1, y1, x2, y2]
#     scores = predictions['scores']  # Shape: [N]
    
#     # Filter by confidence threshold
#     mask = scores > confidence_threshold
#     boxes = boxes[mask]
#     scores = scores[mask]
    
#     # Apply NMS
#     keep_indices = non_max_suppression(boxes, scores, iou_threshold)
    
#     # Get final predictions
#     final_boxes = boxes[keep_indices]
#     final_scores = scores[keep_indices]
    
#     return {
#         'boxes': final_boxes,
#         'scores': final_scores
#     }

# Initialize NCNN model
ncnn_net = ncnn.Net()
ncnn_net.opt.use_vulkan_compute = False
ncnn_net.load_param("best_ncnn_model/model.ncnn.param")
ncnn_net.load_model("best_ncnn_model/model.ncnn.bin")

cap = cv2.VideoCapture(0)
time.sleep(2)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture frame. Exiting...")
            break

        ncnn_input = preprocess_for_ncnn(frame)
        # logger.debug(f"input shape : {ncnn_input.shape}")
        ncnn_input = np.ascontiguousarray(ncnn_input)
        with ncnn_net.create_extractor() as ex:
            mat_in = ncnn.Mat(ncnn_input)
            ex.input("in0", mat_in)
            ret, ncnn_output = ex.extract("out0")
            if ret != 0:
                logger.error("Failed to extract output from NCNN model")
                break
            ncnn_output = np.array(ncnn_output)
            logger.debug(f"NCNN output shape: {ncnn_output.shape}")
            
            # Process detections
            detections = process_raw_output(ncnn_output)
        # logger.debug(f"Detections: {detections}")
        # logger.debug(f"Detections shape:{ detections.shape}")
        try:
            # Log input processing
            # logger.debug(f"Processing frame {frame_count}")
            # logger.debug(f"Input frame shape: {frame.shape}")
            
            # Log NCNN processing
            logger.debug(f"NCNN input shape: {ncnn_input.shape}")
            logger.debug(f"NCNN output shape: {ncnn_output.shape}")
            logger.debug(f"NCNN output: {ncnn_output}")
            
            # Log detections
            logger.debug(f"Number of detections: {len(detections)}")
            for i, det in enumerate(detections):
                logger.debug(f"Detection {i}: {det}")

        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}", exc_info=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    logger.info("Interrupted by user")

cap.release()
cv2.destroyAllWindows()
ncnn_net.clear()