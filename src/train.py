from models import ImageModels
from data_manager import DataManager
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import os

# usage python train.py


class ModelTrain:
    """_summary_
    """

    def __init__(self, model_name, epochs, batch_size, optimizer,
                 loss, metrics):

        self.create_models_dir()
        self.model_name = model_name
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.image_models = ImageModels()

    def load_data(self):

        data_manager = DataManager()
        training_images, training_labels, validation_images, validation_labels = data_manager.cifar_10()

        return training_images, training_labels, validation_images, validation_labels

    def create_models_dir(self):
        if not os.path.exists("../models"):
            os.makedirs("../models")

        if not os.path.exists("../results"):
            os.makedirs("../results")

    def preprocess(self, training_images, validation_images):

        train_x = self.image_models.preprocess_image_input(
            training_images, model_name=self.model_name)
        valid_x = self.image_models.preprocess_image_input(
            validation_images, model_name=self.model_name)

        return train_x, valid_x

    def plot_metrics(self, history):
        """_summary_

        Args:
            history (_type_): _description_
        """

        # summarize history for accuracy
        plt.figure()
        plt.plot(history.history['accuracy'],
                 linestyle='dashed', marker='o', markersize=10)
        plt.plot(history.history['val_accuracy'],
                 linestyle='dashed', marker='o', markersize=10)
        plt.title(f"{self.model_name} model accuracy")
        plt.grid()
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(
            f"../results/{self.model_name}-accuracy.png", bbox_inches='tight')
        plt.close()

        # summarize history for loss
        plt.figure()
        plt.plot(history.history['loss'],
                 linestyle='dashed', marker='o', markersize=10)
        plt.plot(history.history['val_loss'],
                 linestyle='dashed', marker='o', markersize=10)
        plt.title(f"{self.model_name} model loss")
        plt.grid()
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(
            f"../results/{self.model_name}-loss.png", bbox_inches='tight')
        plt.close()

    def run(self):
        """_summary_
        """

        training_images, training_labels, validation_images, validation_labels = self.load_data()

        train_x, valid_x = self.preprocess(training_images, validation_images)

        model = self.image_models.create_model(self.model_name, optimizer=self.optimizer,
                                               loss=self.loss, metrics=self.metrics)

        print(model.summary())

        filepath = f"../models/{self.model_name}.val_accuracy" + \
            "-{val_accuracy:.4f}.hdf5"

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                              patience=3, min_lr=0.001, cooldown=1),
            
            tf.keras.callbacks.EarlyStopping(
                patience=4, monitor='val_accuracy'),
            
            tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                               monitor='val_accuracy',
                                               mode="max",
                                               save_best_only=True),
            
            tf.keras.callbacks.TensorBoard(log_dir='../logs',  histogram_freq=0,  
                                           write_graph=True, write_images=True),
        ]

        history = model.fit(train_x, training_labels, epochs=self.epochs,
                            validation_data=(valid_x, validation_labels),
                            batch_size=self.batch_size, callbacks=callbacks)

        self.plot_metrics(history)


if __name__ == "__main__":
    
    # models: baseline, vgg19, resnet50, inception_v3, mobilenet_v2
    
    model_train = ModelTrain(model_name="resnet50", epochs=30,
                             batch_size=64, optimizer='adam',
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])
    model_train.run()
