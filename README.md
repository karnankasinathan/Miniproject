## Toxic or Non-Toxic Plant Identification Using CNN
Identifying whether a plant is toxic or non-toxic is vital tion.for public health, agriculture, and environmental safety. This project utilizes CNNs to develop a highly accurate model for classifying plants as toxic or non-toxic, improving the accessibility and precision of plant identification

## About

Machine learning and deep learning to address plant safety by identifying toxic and non-toxic plants. By training models like CNN,  VGG16 on a well-structured dataset, the project provides a user-friendly tool for plant identification, aiming to raise awareness about plant toxicity. This application not only helps in individual plant identification by image or name search but also introduces an innovative camera-based feature to enable real-time recognition. This blend of practical use and advanced modeling has potential applications in botany, public safety, and environmental education, supporting users in making informed decisions about plants in their surroundings.



## Features

Image-Based Toxicity Detection:
The image-based toxicity detection feature enables users to determine whether a plant is toxic or non-toxic by simply uploading its image. Users can access a user-friendly interface to upload an image from their device or capture a photo using the camera. The app processes the uploaded image by resizing, normalizing, and preparing it for analysis to ensure compatibility with the trained deep learning model. Advanced algorithms, such as CNN, ResNet, or VGG16, analyze the image to extract features like texture, color, and shape, and predict whether the plant is toxic or non-toxic.

## Requirements

High-Performance GPU: For training complex deep learning models like ResNet, DarkNet, and VGG16 (especially during development).
Plant Toxicity Dataset: A well-labeled dataset containing images of toxic and non-toxic plants (e.g., 5,000 images each as per your current dataset).
Python (with Libraries): Key libraries include TensorFlow/Keras, PyTorch, OpenCV, NumPy, Pandas, and Scikit-Learn.
IDE (Integrated Development Environment): Such as Jupyter Notebook, Google Colab (for prototyping), or PyCharm.
Web Frameworks (for Web Application): Flask or Django, if deploying as a web application.


## System Architecture
<!--Embed the system architecture diagram as shown below-->

![Screenshot 2024-12-18 220729](https://github.com/user-attachments/assets/c84c6608-7139-4292-ae87-94c211690ac3)

# Implementation
```
# Set up parameters
dataset_dir="/content/plant_classification_dataset/tpc-imgs"

IMAGE_SIZE = (224, 224) # Image size required by most models
BATCH_SIZE = 32
EPOCHS = 10  # Adjust epochs based on performance and dataset size
LEARNING_RATE = 1e-4# Data directories
train_dir = "/content/plant_classification_dataset/tpc-imgs"  # Replace with your train dataset directory
validation_dir = "/content/plant_classification_dataset/tpc-imgs"  # Replace with your validation dataset directory
# Data augmentation and loading
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=IMAGE_SIZE,
                                                    batch_size=BATCH_SIZE, class_mode='binary')
validation_generator = val_datagen.flow_from_directory(validation_dir, target_size=IMAGE_SIZE,
                                                       batch_size=BATCH_SIZE, class_mode='binary')
# Function to build the model with specified base architecture
def build_model(base_model):
    base_model.trainable = False  # Freeze the base model layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model# Choose and build the model (example with VGG16, can change to VGG19, InceptionV3, ResNet50, etc.)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
model = build_model(base_model)# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(train_generator,
                    epochs=20,
                    validation_data=validation_generator,
                    callbacks=[early_stopping])
# Evaluate the model
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")# Unfreeze some layers for fine-tuning (optional for further accuracy improvement)
for layer in base_model.layers[-10:]:  # Unfreezing the last few layers
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE/10),  # Lower learning rate for fine-tuning
              loss='binary_crossentropy',
              metrics=['accuracy'])# Fine-tuning
fine_tune_history = model.fit(train_generator,
                              epochs=EPOCHS,
                              validation_data=validation_generator,
                              callbacks=[early_stopping])
                              
# Final evaluation
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Final Validation Accuracy after Fine-Tuning: {val_accuracy * 100:.2f}%")# Import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix# Set up test data directory and image size
test_dir = '/content/plant_classification_dataset/tpc-imgs'  # Replace with your actual test dataset directory
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32# Load the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False  # Important for consistent evaluation
)
# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
# Classification Report
true_classes = test_generator.classes  # True labels
class_labels = list(test_generator.class_indices.keys())  # Class labels
print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

```
## Output

<!--Embed the Output picture at respective places as shown below as shown below-->
#### Output1 

![Screenshot 2024-12-18 221153](https://github.com/user-attachments/assets/2a422b3f-f41a-427d-88b1-9bd548e9c3dd)

![Screenshot 2024-12-18 221449](https://github.com/user-attachments/assets/a386d24f-53f7-4143-a630-8e1ff2134152)


#### Output2 

![Screenshot 2024-12-18 222008](https://github.com/user-attachments/assets/f40fe197-7a46-4207-860e-1062e1ec77f8)



## Results and Impact

Development of a CNN-based model for detecting toxic and non-toxicplants offers a powerful and efficient solution for identifying plant toxicity from images. By leveraging advanced deep learning techniques, the model can automatically extract features from plant images, making accurate predictions about a plant’s toxicity. This system has the potential to be widely applied in various fields such as agriculture, botany, and environmental protection, helping users to avoid harmful plants and ensuring safety in natural and cultivatedenvironments.future enhancements of the toxic and non-toxic plant classification project, severalavenues can be explored to improve the model's accuracy, usability, and applicability. One potential enhancement is to expand the dataset to include a broader variety of plantspecies, particularly those common in different geographical regions, thereby increasingthe model's robustness and generalization capabilities. Additionally, exploring advanceddeep learning architectures, such as ResNet or EfficientNet, could lead to improved performance by leveraging their unique features, such as residual connections and efficient parameter usage.

## Articles published / References

[1] “Deep Learning for Plant Identification in Natural Environment “, YuanLiu, Guan Wang, Haiyan Zhang, 2017
 
[2] “Image Classification for Toxic and Non- Toxic Plants “,MaadShatnawi ,Bakhee ,Almani, 2023

[3] “Plant Toxicity Classificati on by Image “ , Eera Bhatt, 2023

[4] “The Analysis of Plants Image Recognition Based on DeepLearning and ArtificialNeural Network” ,Jiang Huixian ,2020

[5] “What Plant is That? Tests of Automated Image Recognition Apps For Plants Identification On Plants From The British Flora “ , Hamlyn G Jones, 2020

[6] “Automated plant species identification—Trends and future directions”, Jeongyoon HeoJana Waldchen, Michael Rzanny, Marco Seeland, Patrick Mader.,2018

[7] “Plant Species Identification Based on Plant Leaf Using Computer Vision And Machine Learning ”, Prabhpreet Kaur, Surleen Kaur,2023.



