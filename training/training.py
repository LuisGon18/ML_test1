import tensorflow as tf
from tensorflow.keras import layers, models
import logging

# Set up logging to track training progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, input_size: int, num_classes: int):
        self.input_size = input_size
        self.num_classes = num_classes
        self.model = None
        self.history = None

    def build_model(self, learning_rate=0.001):
        """Creates a simple Deep Learning architecture."""
        model = models.Sequential([
            layers.Input(shape=(self.input_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),  # Helps prevent overfitting
            layers.Dense(32, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')  # Softmax for classification
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        logger.info("Model built and compiled successfully.")
        return self.model

    def train(self, x_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """Trains the model with the provided training data."""
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")

        logger.info(f"Starting training for {epochs} epochs...")
        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return self.history

    def save_model(self, file_path="model.h5"):
        """Saves the trained model to a file."""
        if self.model:
            self.model.save(file_path)
            logger.info(f"Model saved to {file_path}")