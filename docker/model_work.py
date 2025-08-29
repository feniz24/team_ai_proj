import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

COLOR_MAP = np.array([
[0, 0, 0],       # Class 0: Black (e.g., background)
[0, 255, 0],     # Class 1: Green
[255, 0, 0],     # Class 2: Red
], dtype=np.uint8)

class ImageModel():
    def __init__(self):
        self.model_path = os.path.join(os.getcwd(), "ml_models", "cat_dog_segmentation_unet.keras")
        self.IMG_HEIGHT = 128
        self.IMG_WIDTH = 128
        self.model = None

    def load_model(self):
        print("Loading model and class indices...")
        try:
            self.model = load_model(self.model_path)
            
        except Exception as e:
            print(f"Error loading model or class indices: {e}")
            return False
 
        return True
    
    def model_predict(self, img):
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, [self.IMG_HEIGHT, self.IMG_WIDTH])
        img_normalized = tf.cast(img, tf.float32) / 255.0

        img_batch = tf.expand_dims(img_normalized, axis=0)

        # Predict
        predicted_masks = self.model.predict(img_batch)
        return predicted_masks , img



    def mask_to_rgb(self, mask, color_map):
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(color_map):
            rgb_mask[mask == class_idx] = color
        return rgb_mask


    def predict_and_overlay(self, img):
        predicted_masks, img = self.model_predict(img)
        pred_mask = tf.argmax(predicted_masks, axis=-1)[0].numpy()

        # Convert mask to color
        color_mask = self.mask_to_rgb(pred_mask, COLOR_MAP)

        # Convert original image tensor for display
        # input_image_numpy = tf.keras.utils.img_to_array(img, dtype=np.uint8)
        input_image_numpy = img.numpy().astype(np.uint8)

        # Blend the input image and the color mask for an overlay effect
        overlay = cv2.addWeighted(input_image_numpy, 0.6, color_mask, 0.4, 0)

        return overlay