import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


TRAIN_DATA_DIR = 'Data/Training' # Path to the training dataset
TEST_DATA_DIR = 'Data/Testing'   # Path to the testing dataset

IMAGE_SIZE = (150, 150) # All images will be resized to this
BATCH_SIZE = 32
EPOCHS = 20 # You can increase this for better training, but it will take longer

# Based on dataset structure (glioma, meningioma, notumor, pituitary)
NUM_CLASSES = 4

# --- 1. Data Preprocessing and Augmentation ---
print("--- Setting up Data Generators ---")

# ImageDataGenerator for training data with augmentation
# Rescale pixel values to [0, 1]
# Apply various augmentations to help the model generalize better
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2 # Use 20% of the training data for validation from TRAIN_DATA_DIR
)

# ImageDataGenerator for test data (only rescaling, no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow images from directory for training
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Changed to 'categorical' for multi-class classification
    subset='training',        # Specify this is the training subset
    seed=42                   # For reproducibility
)

# Flow images from directory for validation
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Changed to 'categorical' for multi-class classification
    subset='validation',      # Specify this is the validation subset
    seed=42                   # For reproducibility
)

# Flow images from directory for testing (separate test set)
test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Changed to 'categorical' for multi-class classification
    shuffle=False,            # Important for evaluation to keep order of predictions consistent with true labels
    seed=42                   # For reproducibility
)

# Get class labels from the training generator (they should be consistent across all)
class_labels = list(train_generator.class_indices.keys())
print(f"Class Labels: {class_labels}")

# --- 2. Build the CNN Model ---
print("\n--- Building the CNN Model ---")

model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25), # Added Dropout for regularization

    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Flatten the 3D output to 1D
    Flatten(),

    # Fully Connected Layers
    Dense(256, activation='relu'),
    Dropout(0.5), # Increased Dropout for the dense layer
    Dense(NUM_CLASSES, activation='softmax') # Changed to NUM_CLASSES units with 'softmax' for multi-class
])

# --- 3. Compile the Model ---
print("\n--- Compiling the Model ---")
model.compile(optimizer=Adam(learning_rate=0.0001), # Using Adam optimizer with a slightly lower learning rate
              loss='categorical_crossentropy', # Changed to 'categorical_crossentropy' for multi-class
              metrics=['accuracy'])

model.summary() # Print a summary of the model architecture

# --- 4. Train the Model ---
print("\n--- Training the Model ---")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE, # Number of batches per epoch
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE # Number of validation batches
)

# --- 5. Evaluate the Model ---
print("\n--- Evaluating the Model on Test Set ---")

# Reset test_generator to ensure consistent order for predictions
test_generator.reset()
Y_pred = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + 1)
# Convert probabilities to class indices (0, 1, 2, 3)
y_pred_classes = np.argmax(Y_pred, axis=1)
# True labels are directly available from the generator
y_true = test_generator.classes[test_generator.index_array]

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# --- 6. Plot Training History ---
print("\n--- Plotting Training History ---")
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

 
print("\n--- Saving the Model ---")
model.save('brain_tumor_cnn_model.h5')
print("Model saved as brain_tumor_cnn_model.h5")

print("\n--- Script Finished ---")
