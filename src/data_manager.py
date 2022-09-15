import tensorflow as tf


class DataManager:
    """_summary_
    """
    def __init__(self):
        self.labels = {
            0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
        }
    
    def cifar_10(self):
        """_summary_
        Ten classes dataset: 'airplane', 'automobile', 'bird', 'cat', 'deer',
                             'dog', 'frog', 'horse', 'ship', 'truck'
        Returns:
            ndarray: training_images, training_labels, validation_images, validation_labels
        """
        
        (training_images, training_labels) , (validation_images, validation_labels) = tf.keras.datasets.cifar10.load_data()
        
        return training_images, training_labels, validation_images, validation_labels

    