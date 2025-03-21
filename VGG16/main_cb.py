import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

# --- Enable GPU --- 
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# --- Base Directory of Project ---
base_dir = os.path.dirname(os.path.abspath(__file__))

# --- Note: make sure not to use the same mouse for testing and validation ---

# --- Paths to Training and Testing datasets ---
training_datasets = {
    "pa_750": os.path.join(base_dir, "Path_Hidden", "750", "training"),
    "pa_850": os.path.join(base_dir, "Path_Hidden", "850", "training"),
    "us_750": os.path.join(base_dir, "Path_Hidden", "750", "training"),
    "us_850": os.path.join(base_dir, "Path_Hidden", "850", "training"),
}

testing_datasets = {
    "pa_750": os.path.join(base_dir, "Path_Hidden", "750", "testing"),
    "pa_850": os.path.join(base_dir, "Path_Hidden", "850", "testing"),
    "us_750": os.path.join(base_dir, "Path_Hidden", "750", "testing"),
    "us_850": os.path.join(base_dir, "Path_Hidden", "850", "testing"),
}

# --- Select dataset path ---
dataset = "pa_750"
training_path = training_datasets[dataset]
testing_path = testing_datasets[dataset]

print(f"Using Training path: {training_path} and Testing path: {testing_path}")

# --- Load datasets ---
training_dataset = tf.keras.utils.image_dataset_from_directory(
    training_path,
    labels='inferred',
    image_size=(224, 224),
    batch_size=16,
    seed=123
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    testing_path,
    labels='inferred',
    image_size=(224, 224),
    batch_size=16,
    seed=123
)

# --- Class names for labels ---
class_names = ['Severe', 'Sham', 'Mild']

# --- Use VGG16 for feature extraction ---
feature_extractor = tf.keras.applications.VGG16(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
feature_extractor.trainable = False

# Function to extract features and labels from a dataset
def extract_features_and_labels(dataset, feature_extractor):
    features = []
    labels = []
    for image_batch, label_batch in dataset:
        extracted_features = feature_extractor.predict(image_batch)
        flattened_features = extracted_features.reshape(extracted_features.shape[0], -1)
        features.append(flattened_features)
        labels.extend(label_batch.numpy())
    return np.vstack(features), np.array(labels)

# --- Extract features and labels for training and validation datasets ---
train_features, train_labels = extract_features_and_labels(training_dataset, feature_extractor)
val_features, val_labels = extract_features_and_labels(validation_dataset, feature_extractor)

# --- Handle class imbalance ---
class_weights = {0: 4.8, 1: 2.5, 2: 2.55}

# --- Train the CatBoost Classifier ---
catboost_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    verbose=100,
    class_weights=class_weights,
    early_stopping_rounds=50
)

# --- Fit the model on extracted features ---
catboost_model.fit(train_features, train_labels, eval_set=(val_features, val_labels))

# --- Generate predictions on the validation set ---
val_predictions = catboost_model.predict(val_features)

# --- Confusion Matrix Plot---
cm = confusion_matrix(val_labels, val_predictions, labels=[0, 1, 2])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - CatBoost')
plt.show()

# --- Visualize feature importance ---
feature_importances = catboost_model.get_feature_importance()
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.xlabel("Feature Index")
plt.ylabel("Importance Score")
plt.title("Feature Importance - CatBoost")
plt.show()
