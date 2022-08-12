from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.EfficientNetB3 import EfficientNetB3

from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionv3
from tensorflow.keras.applications.EfficientNetB3 import preprocess_input as preprocess_input_efficientnet


class ImageModels:
    def __init__(self):
        pass

    def select_model(self, model_name):
        # more models in https://keras.io/api/applications/
        models = {
            # Very Deep Convolutional Networks for Large-Scale Image Recognition (2014)
            "vgg19": VGG19,

            # Deep Residual Learning for Image Recognition (2015)
            "resnet50": ResNet50,

            # Rethinking the Inception Architecture for Computer Vision (2015)
            "inception_v3": InceptionV3,

            # EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (2019)
            "efficientnet": EfficientNetB3
        }
        return models[model_name]
    
    def _preprocess_input(self, model_name):
      
        preprocess = {
            "vgg19": preprocess_input_vgg19,
            "resnet50": preprocess_input_resnet50,
            "inception_v3": preprocess_input_inceptionv3,
            "efficientnet": preprocess_input_efficientnet
        }
        return preprocess[model_name]
    
    def preprocess_image_input(self, input_images,  model_name):
        input_images = input_images.astype('float32')
        output_ims = self._preprocess_input[model_name](input_images)
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

        feature_extractor = self.select_model(model_name)(input_shape=input_shape,
                                                          include_top=False,
                                                          weights='imagenet')(inputs)
        return feature_extractor

    def classifier_layer(self, inputs):
        """_summary_
            Defines final dense layers and subsequent softmax layer for classification.
        """
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
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

    def create_model(self, model_name="resnet50", optimizer='SGD',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy']):

        inputs = tf.keras.layers.Input(shape=(32, 32, 3))

        model = self.model_architecture(inputs, model_name, input_shape=(224, 224, 3))

        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics)

        return model