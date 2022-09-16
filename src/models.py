import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobilenetv2


class ImageModels:
    def __init__(self):
        
        self._models_dict = {
            # baseline
            "baseline": self.baseline_model,
        
            # Very Deep Convolutional Networks for Large-Scale Image Recognition (2014)
            "vgg19": VGG19,

            # Deep Residual Learning for Image Recognition (2015)
            "resnet50": ResNet50,

            # Rethinking the Inception Architecture for Computer Vision (2015)
            "inception_v3": InceptionV3,
            
            # MobileNetV2: Inverted Residuals and Linear Bottlenecks (2018)
            "mobilenet_v2": MobileNetV2
        }
    
    def preprocess_baseline(self, images):
        return images / 255  
           
    def baseline_model(self, inputs):
        
        x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3))(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        x = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(
            10, activation="softmax", name="classification")(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=x)
    
        return model
    
    def list_models(self):
        return list(self._models_dict.keys())

    def select_model(self, model_name):
        # more models in https://keras.io/api/applications/
        return self._models_dict[model_name]
    
    def _preprocess_input(self, model_name):
      
        preprocess = {
            "baseline": self.preprocess_baseline,
            "vgg19": preprocess_input_vgg19,
            "resnet50": preprocess_input_resnet50,
            "inception_v3": preprocess_input_inceptionv3,
            "mobilenet_v2": preprocess_input_mobilenetv2,
        }
        return preprocess[model_name]
    
    def preprocess_image_input(self, input_images,  model_name):
        input_images = input_images.astype('float32')
        output_ims = self._preprocess_input(model_name)(input_images)
        return output_ims

    def feature_extractor(self, inputs, model_name, input_shape):
        """_summary_
            Feature Extraction is performed with pretrained models on imagenet weights. 
            Input size is 224 x 224.
        Args:
            inputs any: _description_

        Returns:
            feature_extractor
        """

        pretrained_model = self.select_model(model_name)(input_shape=input_shape,
                                                          include_top=False,
                                                          weights='imagenet')
        pretrained_model.trainable = False
        pretrained_model = pretrained_model(inputs)
        
        return pretrained_model

    def classifier_layer(self, inputs):
        """_summary_
            Defines final dense layers and subsequent softmax layer for classification.
        """
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(
            10, activation="softmax", name="classification")(x)
        return x

    def model_architecture(self, inputs, model_name, input_shape):
        """_summary_
            Build the model using transfer learning
        Args:
            inputs (_type_): _description_

        Returns:
            model
        """

        resize = tf.keras.layers.UpSampling2D(size=(7, 7))(inputs)

        feature_extractor = self.feature_extractor(resize, model_name, input_shape)

        classification_output = self.classifier_layer(feature_extractor)

        model = tf.keras.Model(inputs=inputs, outputs=classification_output)

        return model

    def create_model(self, model_name, optimizer, loss, metrics):

        inputs = tf.keras.layers.Input(shape=(32, 32, 3))
        
        if model_name == "baseline":
            model = self._models_dict["baseline"](inputs)
        else:
            model = self.model_architecture(inputs, model_name, input_shape=(224, 224, 3))

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        return model