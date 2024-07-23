#import libraries 
import tensorflow as tf  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  
from tensorflow.keras.preprocessing.image import ImageDataGenerator  
from tensorflow.keras.optimizers import Adam  
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau  
from tensorflow.keras.utils import plot_model  
import numpy as np  
import cv2  
import os  
import glob  
import random  
import matplotlib.pyplot as plt  
import datetime
  
# Define constants  
IMG_SIZE = 224  
BATCH_SIZE = 32  
NUM_CLASSES = 26  # Number of sign language labels  
EPOCHS = 10  
  
# Load the dataset  
train_dir = 'asl_dataset/train'  
test_dir = 'asl_dataset/test'  
  
train_datagen = ImageDataGenerator(rescale=1./255)  
test_datagen = ImageDataGenerator(rescale=1./255)
sign_language_labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]  
  
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')  
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')  
  
# Define the model architecture  
model = Sequential()  
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))  
model.add(MaxPooling2D((2, 2)))  
model.add(Conv2D(64, (3, 3), activation='relu'))  
model.add(MaxPooling2D((2, 2)))  
model.add(Conv2D(128, (3, 3), activation='relu'))  
model.add(MaxPooling2D((2, 2)))  
model.add(Flatten())  
model.add(Dense(128, activation='relu'))  
model.add(Dense(NUM_CLASSES, activation='softmax'))  
  
# Compile the model  
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])  
  
# Define callbacks  
checkpoint = ModelCheckpoint('sign_language_model.weights.h5', monitor='val_accuracy', save_weights_only=True, mode='max', verbose=1)  
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=0.001)  
  
# Train the model  
history = model.fit(train_generator, epochs=EPOCHS, validation_data=test_generator, callbacks=[checkpoint, reduce_lr])  
  
# Evaluate the model  
model.evaluate(test_generator)  
  
# Save the model  
model.save('sign_language_model.h5')  
  
# Load the model for deployment  
model = tf.keras.models.load_model('sign_language_model.h5')  
  
# Define a function to predict sign language from an image  
def predict_sign_language(image_path):  
    img = cv2.imread(image_path)  
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    pred = model.predict(img)  
    return np.argmax(pred)  
  
# Create a GUI application using Tkinter  
import tkinter as tk  
from tkinter import filedialog  
from PIL import Image, ImageTk  

# Check if the local time is b/w 6 PM to 10 PM
local_time = datetime.datetime.now()
curr_time = local_time.strftime("%H:%M:%S")
if curr_time >= "18:00:00" and curr_time <="22:00:00":
    def upload_image():  
        global file_path  
        file_path = filedialog.askopenfilename()  
        uploaded = Image.open(file_path)  
        uploaded.thumbnail((224, 224))  
        im = ImageTk.PhotoImage(uploaded)  
        sign_image.configure(image=im)  
        sign_image.image = im  
        label1.configure(text='')  
    
    def detect_sign_language():  
        pred = predict_sign_language(file_path)  
        label1.configure(text=sign_language_labels[pred])  
    
    root = tk.Tk()  
    root.title('Sign Language Recognition')  
    root.geometry('400x400')  
    
    upload = tk.Button(root, text='Upload Image', command=upload_image)  
    upload.pack()  
    
    sign_image = tk.Label(root)  
    sign_image.pack()  
    label1 = tk.Label(root, text='')  
    label1.pack()  
    
    detect_button = tk.Button(root, text='Detect Sign Language', command=detect_sign_language)  
    detect_button.pack()  
    root.mainloop() 

else:
    window = tk.Tk()
    lbl_no = tk.Label(window, text='This Model is trained to Predict Sign Language only between 6PM and 10PM. Please try again later!', font='Arial 15').pack(pady=15)
    window.mainloop()