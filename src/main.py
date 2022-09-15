from keras.models import load_model
from utils import load_and_preprocess_image, predict
    

if __name__ == "__main__":
    
    MODEL_NAME = "mobilenet_v2"
    
    image = load_and_preprocess_image("predict_airplane.png", MODEL_NAME)
    
    # look in the models dir
    model = load_model('../models/mobilenet_v2.val_accuracy-0.7311.hdf5')
    
    result = predict(model, image)
    
    print("Predicted class:", result)