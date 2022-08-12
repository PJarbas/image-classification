from models import ImageModels
from data_manager import DataManager

if __name__ == "__main__":
    
    # get cifar-10 data
    data = DataManager()
    training_images, training_labels, validation_images, validation_labels = data.cifar_10()
    
    # preprocess
    image_models = ImageModels()
    
    train_X = image_models.preprocess_image_input(training_images, model_name="resnet50")
    valid_X = image_models.preprocess_image_input(validation_images, model_name="resnet50")
    
    print(valid_X.shape)
    
