import tensorflow as tf

class DataManager:
    def __init__(self):
        pass
    
    def cifar_10(self):
        """_summary_
        Ten classes dataset: 'airplane', 'automobile', 'bird', 'cat', 'deer',
                             'dog', 'frog', 'horse', 'ship', 'truck'
        Returns:
            ndarray: training_images, training_labels, validation_images, validation_labels
        """
        
        (training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()
        
        return training_images, training_labels, validation_images, validation_labels 
    