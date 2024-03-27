import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


# Paths
project_dir = '/Users/siddhantraj/dev/Speech_Recognition_Project'
data_dir = os.path.join(project_dir, 'audioTest_Data')
transcription_file = os.path.join(data_dir, 'transcription.txt')

# Parameters
n_mfcc = 40
max_pad_length = 400  # Adjust based on your data

def extract_mfcc(file_path, max_pad_length):
    # audio, sample_rate = librosa.load(file_path, sr=None)
    # mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    # pad_width = max_pad_len - mfccs.shape[1]
    # mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # return mfccs
    
    audio, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Truncate MFCCs if their length exceeds the maximum pad length
    if mfccs.shape[1] > max_pad_length:
        mfccs = mfccs[:, :max_pad_length]

    # Calculate the pad width based on the desired length
    pad_width = max_pad_length - mfccs.shape[1]

    # Pad the MFCCs to the desired length
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return mfccs


def load_and_preprocess_data(data_dir, transcription_file):
    X, y = [], []
    with open(transcription_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            audio_path = os.path.join(data_dir, parts[0] + '.wav')  # Adjust based on actual file names/extensions
            transcript = parts[1]
            mfcc = extract_mfcc(audio_path, max_pad_length)
            X.append(mfcc.T)  # Transpose to align with expected input shape
            y.append(transcript)
    return np.array(X), np.array(y)
# def load_and_preprocess_data(data_dir, transcription_file):
#     X, y = [], []
#     with open(transcription_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split(' ', 1)
#             if len(parts) != 2:
#                 print("Error: Unexpected line format:", line)
#                 continue
#             audio_path = os.path.join(data_dir, parts[0] + '.wav')  # Adjust based on actual file names/extensions
#             transcript = parts[1]
#             mfcc = extract_mfcc(audio_path, max_pad_length)
#             X.append(mfcc.T)  # Transpose to align with expected input shape
#             y.append(transcript)
#     return np.array(X), np.array(y)

# Load and preprocess data
X, y = load_and_preprocess_data(data_dir, transcription_file)

# Encode text data
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train_categorical = to_categorical(y_train, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_pad_length, n_mfcc)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")



# import os
# import numpy as np
# import librosa
# import tensorflow as tf
# from tensorflow import keras
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from keras.utils import to_categorical
# from keras.callbacks import LearningRateScheduler
# from keras.layers import Dropout

# # Paths
# project_dir = '/Users/siddhantraj/dev/Speech_Recognition_Project'
# data_dir = os.path.join(project_dir, 'audioTest_Data')
# transcription_file = os.path.join(data_dir, 'transcription.txt')

# # Parameters
# n_mfcc = 40
# max_pad_length = 400  # Adjust based on your data

# def extract_mfcc(file_path, max_pad_length):
#     audio, sample_rate = librosa.load(file_path)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

#     # Truncate MFCCs if their length exceeds the maximum pad length
#     if mfccs.shape[1] > max_pad_length:
#         mfccs = mfccs[:, :max_pad_length]

#     # Calculate the pad width based on the desired length
#     pad_width = max_pad_length - mfccs.shape[1]

#     # Pad the MFCCs to the desired length
#     mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

#     return mfccs

# def load_and_preprocess_data(data_dir, transcription_file):
#     X, y = [], []
#     with open(transcription_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split(' ', 1)
#             audio_path = os.path.join(data_dir, parts[0] + '.wav')  # Adjust based on actual file names/extensions
#             transcript = parts[1]
#             mfcc = extract_mfcc(audio_path, max_pad_length)
#             X.append(mfcc.T)  # Transpose to align with expected input shape
#             y.append(transcript)
#     return np.array(X), np.array(y)

# # Load and preprocess data
# X, y = load_and_preprocess_data(data_dir, transcription_file)

# # Encode text data
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# num_classes = len(label_encoder.classes_)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# # Convert labels to categorical
# y_train_categorical = to_categorical(y_train, num_classes=num_classes)
# y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# # Build the model with increased complexity and regularization
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(max_pad_length, n_mfcc)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
#     tf.keras.layers.Dropout(0.2),  # Option 2: Regularization (Add dropout layer)
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
#     tf.keras.layers.Dropout(0.2),  # Option 2: Regularization (Add dropout layer)
#     tf.keras.layers.Dense(128, activation='relu'),  # Option 1: Increase model complexity (Add additional dense layer)
#     tf.keras.layers.Dropout(0.2),  # Option 2: Regularization (Add dropout layer)
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])

# # Learning rate scheduler function for option 3
# def lr_scheduler(epoch, lr):
#     if epoch % 5 == 0:  # Decrease learning rate every 5 epochs
#         lr *= 0.9  # Adjust the multiplication factor as needed
#     return lr

# # Compile the model with learning rate scheduler
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Set initial learning rate
#               loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model with learning rate scheduler
# model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, validation_split=0.2, 
#           callbacks=[LearningRateScheduler(lr_scheduler)])

# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
# print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# import os
# import numpy as np
# import librosa
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from keras.utils import to_categorical
# from keras.applications import VGG16
# from keras.layers import Input, Flatten, Dense
# import ssl

# # Disable SSL certificate verification
# ssl._create_default_https_context = ssl._create_unverified_context

# # Paths
# project_dir = '/Users/siddhantraj/dev/Speech_Recognition_Project'
# data_dir = os.path.join(project_dir, 'audioTest_Data')
# transcription_file = os.path.join(data_dir, '84-121123.trans.txt')

# # Parameters
# n_mfcc = 40
# max_pad_length = 400  # Adjust based on your data

# def extract_mfcc(file_path, max_pad_length):
#     audio, sample_rate = librosa.load(file_path)
#     mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

#     # Truncate MFCCs if their length exceeds the maximum pad length
#     if mfccs.shape[1] > max_pad_length:
#         mfccs = mfccs[:, :max_pad_length]

#     # Calculate the pad width based on the desired length
#     pad_width = max_pad_length - mfccs.shape[1]

#     # Pad the MFCCs to the desired length
#     mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

#     return mfccs

# def load_and_preprocess_data(data_dir, transcription_file):
#     X, y = [], []
#     with open(transcription_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split(' ', 1)
#             audio_path = os.path.join(data_dir, parts[0] + '.wav')  # Adjust based on actual file names/extensions
#             transcript = parts[1]
#             mfcc = extract_mfcc(audio_path, max_pad_length)
#             X.append(mfcc.T)  # Transpose to align with expected input shape
#             y.append(transcript)
#     return np.array(X), np.array(y)

# # Load and preprocess data
# X, y = load_and_preprocess_data(data_dir, transcription_file)

# # Encode text data
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# num_classes = len(label_encoder.classes_)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# # Load and preprocess data with three channels
# X_train = np.stack((X_train,) * 3, axis=-1)  # Duplicate the single channel along the last axis
# X_test = np.stack((X_test,) * 3, axis=-1)  # Duplicate the single channel along the last axis
# # Convert labels to categorical
# y_train_categorical = to_categorical(y_train, num_classes=num_classes)
# y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# # Load pre-trained VGG16 model without the top (fully connected) layers
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(max_pad_length, n_mfcc, 3))

# # Freeze the base model layers
# base_model.trainable = False

# # Add custom top layers for the classification task
# inputs = Input(shape=(max_pad_length, n_mfcc, 3))
# x = base_model(inputs, training=False)
# x = Flatten()(x)
# x = Dense(128, activation='relu')(x)
# outputs = Dense(num_classes, activation='softmax')(x)
# model = tf.keras.Model(inputs, outputs)

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, validation_split=0.2)

# # Evaluate the model
# test_loss, test_accuracy = model.evaluate(X_test, y_test_categorical)
# print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
