from re import S
import numpy as np
import h5py
import matplotlib.pyplot as plt
from DLHandsighns import *
import pandas as pd
from PIL import Image, ImageEnhance
import os


g_total_num_classes = 26
g_subset_num_classes = 12
g_weights_file = r"C:\Users\User\Downloads\Dataset_signLetters\Project_YA_mika\Project_YA_mika\Project_YA_mika_Weights\weights_Mika"
g_load_weights_from_file =  True # If True, the weights will be loaded from the file. If False, the weights will be trained and saved to the file
g_num_px = 28
g_num_data = 19600 # Number of data images for the training

 
# Function to find the path of a folder in a directory
def find_folder_path(start_dir, target_folder_name, required_subfolders=None):
    for root, dirs, files in os.walk(start_dir):
        if target_folder_name in dirs:
            potential_path = os.path.join(root, target_folder_name)
            
            # Check if the required subfolders exist in the found folder
            if required_subfolders:
                if all(os.path.exists(os.path.join(potential_path, subfolder)) for subfolder in required_subfolders):
                    return potential_path
            else:
                return potential_path
    return None


# Start searching from the home directory
base_search_dir = os.path.expanduser('~')  # Start from the home directory for cross-platform compatibility

# Define required subfolders to validate the correct 'Project_YA_mika' folder
required_subfolders = ["Dataset_Mika", "Project_YA_mika_Weights"]

# Find the 'Project_YA_mika' folder path
project_folder_path = find_folder_path(base_search_dir, "Project_YA_mika", required_subfolders)

if project_folder_path:
    # Construct the paths to the dataset files and weights inside the located 'Project_YA_mika' folder
    train_data_path = os.path.join(project_folder_path, "Dataset_Mika", "sign_mnist_train.csv")
    test_data_path = os.path.join(project_folder_path, "Dataset_Mika", "sign_mnist_test.csv")
    
    # Construct the paths to the weights files inside the 'weights_Mika' folder
    weights_folder_path = os.path.join(project_folder_path, "Project_YA_mika_Weights", "weights_Mika")
    layer1_path = os.path.join(weights_folder_path, "Layer1.h5")
    layer2_path = os.path.join(weights_folder_path, "Layer2.h5")

    # Load the CSV files with explicit encoding
    try:
        train_data = pd.read_csv(train_data_path, encoding='utf-8').dropna().apply(pd.to_numeric, errors='coerce')
        test_data = pd.read_csv(test_data_path, encoding='utf-8').dropna().apply(pd.to_numeric, errors='coerce')
        
        print("Training data loaded successfully.")
        print(f"Shape of training data: {train_data.shape}")
        
        print("Test data loaded successfully.")
        print(f"Shape of test data: {test_data.shape}")
        
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
    
    # Check if weights files exist
    if os.path.exists(layer1_path):
        print("Layer 1 weights file found successfully.")
    else:
        print("Layer 1 weights file not found. Please check the path.")

    if os.path.exists(layer2_path):
        print("Layer 2 weights file found successfully.")
    else:
        print("Layer 2 weights file not found. Please check the path.")
else:
    print("Project_YA_mika folder not found with the required structure. Please ensure it exists and has the necessary subfolders.")
    

# creating the model    
def create_signs_model():
    
    # creating the layers
    if g_load_weights_from_file == False:
        hidden_layer1 = DLLayer ("l1", 200,(784,),"relu","He", 1)
        softmax_layer = DLLayer ("l2", 26,(200,),"softmax","He", 1)
    else:
        hidden_layer1 = DLLayer ("l1", 200,(784,),"relu", W_initialization = layer1_path, learning_rate = 1)
        softmax_layer = DLLayer ("l2", 26,(200,),"softmax",W_initialization = layer2_path, learning_rate = 1)
    
    # creating the model, using the layers we previously created    
    model = DLModel()
    model.add(hidden_layer1)
    model.add(softmax_layer)
    model.compile("categorical_cross_entropy")

    return model

# converts a vector Y of class labels into a one-hot encoded matrix, handling a subset of classes if specified.    
def Y_to_one_hot(Y):
    one_hot_Y = np.zeros((Y.size, g_total_num_classes))
    if g_subset_num_classes != 0:
        mask = np.isin(Y, np.arange(g_subset_num_classes))
        indices = np.where(mask, Y, g_subset_num_classes)
        indices = np.clip(indices, 0, g_total_num_classes - 1)
        one_hot_Y[np.arange(Y.size), indices] = 1
    else:
        one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

# Function to subset the dataset to a specified number of classes
def subset_dataset(X, Y, allowed_labels):
    mask = np.isin(Y, np.arange(g_subset_num_classes))
    print("Shape of X:", X.shape)
    print("Shape of Y:", Y.shape)

    X = X.T
    filtered_X = X[mask]
    filtered_Y = Y[mask]
    
    return filtered_X.T, filtered_Y

# Function to apply random brightness and contrast adjustment to a normalized image
def random_brightness_contrast(image_array, min_brightness=0.5, max_brightness=1.5, min_contrast=0.5, max_contrast=1.5):
    # Convert normalized array to range [0, 255] for PIL processing
    image_array_255 = ((image_array + 0.5) * 255).astype(np.uint8)
    image = Image.fromarray(image_array_255)
    
    # Apply random brightness
    brightness_enhancer = ImageEnhance.Brightness(image)
    brightness_factor = random.uniform(min_brightness, max_brightness)
    image = brightness_enhancer.enhance(brightness_factor)
    
    # Apply random contrast
    contrast_enhancer = ImageEnhance.Contrast(image)
    contrast_factor = random.uniform(min_contrast, max_contrast)
    image = contrast_enhancer.enhance(contrast_factor)
    
    # Convert back to normalized range [-0.5, 0.5]
    adjusted_image_array = np.array(image).astype(np.float32) / 255.0 - 0.5
    return adjusted_image_array

# Function to apply the brightness and contrast adjustment to a batch of images
def apply_augmentation_to_batch(images_array, min_brightness=0.5, max_brightness=1.5, min_contrast=0.5, max_contrast=1.5):
    # Create an array to hold the augmented images
    adjusted_images_array = np.empty_like(images_array)
    
    # Apply the augmentation to each image
    for i in range(images_array.shape[1]):
        image_array_flat = images_array[:, i]
        image_array_2d = image_array_flat.reshape((28, 28))
        adjusted_image_array_2d = random_brightness_contrast(image_array_2d, min_brightness, max_brightness, min_contrast, max_contrast)
        adjusted_images_array[:, i] = adjusted_image_array_2d.flatten()
    
    return adjusted_images_array


def main():  
    print (f"Number of classes: {g_subset_num_classes}")
    curr_model = create_signs_model()
    curr_model.forced_num_classess = g_subset_num_classes

#--- Prepare the data for training
 

    X_train = train_data.iloc[:, 1:].values.T
    X_train = X_train / 255.0 -0.5
    Y = train_data.iloc[:, 0].values.T
    classes = list(range(9)) + [10, 11, 12] # Num 9 is the letter J, in sign language it requires movement, so it was removed
    X_train, Y = subset_dataset(X_train, Y, classes)
    X_train = apply_augmentation_to_batch(X_train)
    Y_train = Y_to_one_hot(Y)\
    
    # Apply the shuffled indices to the data            
    shuffled_indices = np.random.permutation(Y.size)
    shuffled_Y_Train = Y_train[:, shuffled_indices]
    shuffled_X_Train = X_train[:, shuffled_indices]
            
    #--- Train the model
    if g_load_weights_from_file == False:
        costs = curr_model.train(X_train,Y_train,g_num_data)
        curr_model.save_weights(g_weights_file)
        print(costs)
    
    #--- Prepare the data for testing
    X_test = test_data.iloc[:, 1:].values.T
    X_test = X_test / 255.0 -0.5
    Y = test_data.iloc[:, 0].values.T
    X_test, Y = subset_dataset(X_test, Y, classes)
    Y_test = Y_to_one_hot(Y)
    
    #--- Test the model
    print('Deep train accuracy')
    curr_model.confusion_matrix(X_train, Y_train)
    print('Deep test accuracy')
    curr_model.confusion_matrix(X_test, Y_test)
   
   
   #--- Predict a self made image   
    img_path = r"Enter path here" # Enter the path of the image you want to predict
    img = Image.open(img_path)
    
    # convert image to grayscale
    img_gray = img.convert('L')

    img_gray_np = np.array(img_gray)
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')  
    plt.show()
    
    image64 = img_gray.resize((g_num_px, g_num_px), Image.LANCZOS)
    plt.imshow(img_gray)
    plt.show()
    plt.imshow(image64)
    plt.show();
    
    my_image = np.reshape(image64,(g_num_px*g_num_px,1))
    my_image = my_image/255. - 0.5
    p = curr_model.predict(my_image)
    prediction_index = np.argmax(p, axis=0)
    print (prediction_index.item())
        
    
if __name__ == "__main__":
    main()