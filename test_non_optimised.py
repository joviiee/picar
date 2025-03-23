import cv2
import time 
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Multiply, Conv2D, Concatenate,Lambda
print("import done")

class ChannelAttention(Layer):
    def __init__(self, channels, reduction=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = GlobalAveragePooling2D()
        self.max_pool = GlobalMaxPooling2D()
        self.fc1 = Dense(channels // reduction, activation="relu", use_bias=False)
        self.fc2 = Dense(channels, activation="sigmoid", use_bias=False)

    def call(self, inputs):
        avg_out = self.fc2(self.fc1(self.avg_pool(inputs)))
        max_out = self.fc2(self.fc1(self.max_pool(inputs)))
        out = avg_out + max_out
        return Multiply()([inputs, out])

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            "channels": self.channels,
            "reduction": self.reduction
        })
        return config
class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv = Conv2D(1, kernel_size=7, strides=1, padding="same", activation="sigmoid", use_bias=False)

    def call(self, inputs):
        avg_out = Lambda(lambda x: keras.backend.mean(x, axis=-1, keepdims=True))(inputs)
        max_out = Lambda(lambda x: keras.backend.max(x, axis=-1, keepdims=True))(inputs)
        concat = Concatenate(axis=-1)([avg_out, max_out])
        attention = self.conv(concat)
        return Multiply()([inputs, attention])

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        return config
    
custom_objects = {
    "ChannelAttention": ChannelAttention,
    "SpatialAttention": SpatialAttention
}

yolo_model = YOLO("best.pt") 
cnn_model = load_model("best_model.h5", custom_objects=custom_objects)

print("\n\nload success")

label={0:'30km/h',1:'40km/h',2:'60km/h',3:'80km/h',4:'pedestrain',5:'school zone'}

def preprocess_for_cnn(cropped_img):
    resized = cv2.resize(cropped_img, (256, 256)) 
    normalized = resized / 255.0  
    array = img_to_array(normalized)
    return np.expand_dims(array, axis=0)  

cap = cv2.VideoCapture(1)  
time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    results = yolo_model.predict(frame, conf=0.5,verbose=False) 
    detections = results[0].boxes.data.cpu().numpy()  
    print("detection done ")
    for detection in detections:
        x1, y1, x2, y2, conf, cls = map(int, detection[:6])  
        class_id = int(cls)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cropped_img = frame[y1:y2, x1:x2]
        preprocessed_img = preprocess_for_cnn(cropped_img)

        prediction = cnn_model.predict(preprocessed_img)
        
        # print the model confidence
        print(prediction)
        predicted_class = np.argmax(prediction)

        prediction_label = f"class: {predicted_class}"
        print(label[predicted_class])
        cv2.putText(frame, label[predicted_class], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # cv2.imshow("Traffic Speed Sign Detection", frame)
    cv2.imwrite("img.jpg",frame)
    print("image displayed")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
