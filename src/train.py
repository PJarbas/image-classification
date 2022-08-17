from models import ImageModels
from data_manager import DataManager
from matplotlib import pyplot as plt
import numpy as np


class ModelTrain:
    def __init__(self, model_name, epochs, batch_size, optimizer,
                     loss, metrics):
        
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
    
    def preprocess(self, training_images, validation_images):
        
        train_x = self.image_models.preprocess_image_input(training_images, model_name=self.model_name)
        valid_x = self.image_models.preprocess_image_input(validation_images, model_name=self.model_name)
        
        return train_x, valid_x
    
    def plot_metrics(history):
        
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.grid()
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("accuracy.png", bbox_inches='tight')
        
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.grid()
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("loss.png", bbox_inches='tight')
        
        
    
    def run(self):
        
        training_images, training_labels, validation_images, validation_labels = self.load_data()
        
        train_x, valid_x = self.preprocess(training_images, validation_images)
        
        model = self.image_models.create_model(self.model_name, optimizer=self.optimizer,
                     loss=self.loss, metrics=self.metrics)
        
        print(model.summary())
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=3, monitor="loss"),
            tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        ]
        
        history = model.fit(train_x, training_labels, epochs=self.epochs,
                            validation_data = (valid_x, validation_labels),
                            batch_size=self.batch_size, callbacks=callbacks)
        
        model.save(f"../models/{self.model_name}.h5")
        
        loss, accuracy = model.evaluate(valid_x, validation_labels, batch_size=self.batch_size)
        
        self.plot_metrics(history)
        
        # probabilities = model.predict(valid_X, batch_size=64)
        # probabilities = np.argmax(probabilities, axis = 1)

        # display_images(validation_images, probabilities, validation_labels, "Bad predictions indicated in red.")
        

if __name__ == "__main__":
    
    model_train = ModelTrain(model_name="mobilenet_v2", epochs=4,
                             batch_size=64, optimizer='SGD',
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])
    model_train.run()
    
    
