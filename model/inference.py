import onnxruntime as ort
import numpy as np


class ONNXModel:
    def __init__(self, model_path: str, class_names: list):
        self.class_names = class_names

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_tensor: np.ndarray):
        outputs = self.session.run(
            None,
            {self.input_name: input_tensor}
        )

        logits = outputs[0][0]
        probs = self._softmax(logits)

        pred_idx = int(np.argmax(probs))

        return {
            "class_name": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx])
        }

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

