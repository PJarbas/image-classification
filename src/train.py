from ..models import ImageModels
from data_manager import DataManager

if __name__ == "__main__":
    
    # get cifar-10 data
    data = DataManager()
    training_images, training_labels, validation_images, validation_labels = data.cifar_10()
    print(validation_images.shape)
    exit()
    
    # preprocess
    image_models = ImageModels()
    
    train_X = image_models.preprocess_image_input(training_images)
    valid_X = image_models.preprocess_image_input(validation_images)
    
