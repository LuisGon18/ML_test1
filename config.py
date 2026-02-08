import os
from dotenv import load_dotenv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

load_dotenv()

CSV_DELIMITER: str = os.getenv("CSV_DELIMITER")
CSV_ENCODING: str = os.getenv("CSV_ENCODING")