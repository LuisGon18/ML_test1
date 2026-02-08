from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

class PreprocessingData:
    def __init__(self, test_size=0.10, random_state=101):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

        self.x_transform = None
        self.y_encoded = None
        self.y_onehot = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.input_size = None

    def fit_transform(self, x, y):
        # Standardize the features
        self.scaler = StandardScaler().fit(x)
        self.x_transform = self.scaler.transform(x)

        # Encode the target variable
        self.encoder = LabelEncoder()
        self.y_encoded = self.encoder.fit_transform(y)

        # One-hot encode the target labels
        self.y_onehot = tf.keras.utils.to_categorical(self.y_encoded)

        # Split the data into training and testing sets
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_transform,
            self.y_onehot,
            test_size=self.test_size,
            random_state=self.random_state
        )

        # Define the input size for each data sample
        self.input_size = self.x_train.shape[1]

        return self.x_train, self.x_test, self.y_train, self.y_test, self.input_size
