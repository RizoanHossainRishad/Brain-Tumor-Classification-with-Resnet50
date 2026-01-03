import cv2
import numpy as np

def preprocess_image(image_bytes):
    """
    Preprocess uploaded image for ResNet50 ONNX model
    """
    # Convert bytes to numpy array
    np_img = np.frombuffer(image_bytes, np.uint8)

    # Decode image
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)


    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # CHW format
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img
