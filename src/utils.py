import matplotlib.pyplot as plt
import numpy as np
import cv2
from data_manager import DataManager
from keras.models import load_model
from models import ImageModels



# "predict_airplane.png"

def load_and_preprocess_image(image_path, model_name):
    
    image_models = ImageModels()
    
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # resize to cifar10 size
    resized_image = cv2.resize(img, (32, 32))
    
    processed_img = image_models.preprocess_image_input(
            resized_image, model_name=model_name)
    
    return processed_img

def predict(model, image):
    data_manager = DataManager()
    labels_names = data_manager.labels
    
    image = np.expand_dims(image, axis=0)
    
    p = model.predict(image, batch_size=64)
    p = np.argmax(p, axis = 1)[0]
    return labels_names[p]