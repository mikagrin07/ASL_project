from re import S
import numpy as np
import h5py
import matplotlib.pyplot as plt
from DLHandsighns import *
import pandas as pd
from PIL import Image, ImageEnhance
import os
#from unit10 import c1w5_utils as u10



####
current_directory = os.getcwd()
print("###################################################################Current Working Directory:", current_directory)
#####


g_total_num_classes = 26
g_subset_num_classes = 12
g_weights_file =  r"myflaskapp/weights/Project_YA_mika"
g_load_weights_from_file =  True
g_num_px = 28
g_len_data = 19607


def display_image_from_row(data, row_index, additional_label = ""):
    selected_row = data.iloc[row_index, 1:785].values
    image = selected_row.reshape(g_num_px,g_num_px)

    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.title(f"Label: {data.iloc[row_index, 0]}, Additional Label: {additional_label}")
    plt.axis('off')
    plt.show()

def create_signs_model():

    np.random.seed(1)
    if g_load_weights_from_file == False:
        hidden_layer1 = DLLayer ("l1", 200,(784,),"relu","He", 1)
        softmax_layer = DLLayer ("l5", 26,(200,),"softmax","He", 1)
    else:
        hidden_layer1 = DLLayer ("l1", 200,(784,),"relu", W_initialization = f"{g_weights_file}/Layer1.h5", learning_rate = 1)
        softmax_layer = DLLayer ("l5", 26,(200,),"softmax",W_initialization = f"{g_weights_file}/Layer2.h5", learning_rate = 1)

    model = DLModel()
    model.add(hidden_layer1)
    model.add(softmax_layer)
    model.compile("categorical_cross_entropy")

    return model

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

# Function to display an image
def display_image(image_array, title="Image"):
    # Convert normalized array to range [0, 255] for display
    image_array_255 = ((image_array + 0.5) * 255).astype(np.uint8)
    plt.imshow(image_array_255, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    print (f"Number of classes: {g_subset_num_classes}")
    curr_model = create_signs_model()
    curr_model.forced_num_classess = g_subset_num_classes

    #--- Prepare the data for training
    train_data = pd.read_csv(r"C:\Users\User\Downloads\Dataset_signLetters\sign_mnist_train.csv")
    X_train = train_data.iloc[:, 1:].values.T
    X_train = X_train / 255.0 -0.5
    Y = train_data.iloc[:, 0].values.T
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
    X_train, Y = subset_dataset(X_train, Y, classes)
    X_train = apply_augmentation_to_batch(X_train)
    Y_train = Y_to_one_hot(Y)\

    # Apply the shuffled indices to the data
    shuffled_indices = np.random.permutation(Y.size)
    shuffled_Y_Train = Y_train[:, shuffled_indices]
    shuffled_X_Train = X_train[:, shuffled_indices]

    #--- Train the model
    if g_load_weights_from_file == False:
        costs = curr_model.train(X_train,Y_train,19600)
        curr_model.save_weights(g_weights_file)
        print(costs)

    #--- Prepare the data for testing
    test_data = pd.read_csv(r"C:\Users\User\Downloads\Dataset_signLetters\sign_mnist_test.csv")
    small_test_data = test_data.head(8650)
    X_test = small_test_data.iloc[:, 1:].values.T
    X_test = X_test / 255.0 -0.5
    Y = small_test_data.iloc[:, 0].values.T
    X_test, Y = subset_dataset(X_test, Y, classes)
    Y_test = Y_to_one_hot(Y)

    #--- Test the model
    print('Deep train accuracy')
    curr_model.confusion_matrix(X_train, Y_train)
    print('Deep test accuracy')
    curr_model.confusion_matrix(X_test, Y_test)
    g_num_data = g_num_data * 2




    img_path = r"C:\Users\User\Downloads\Dataset_signLetters\LetterMCheck3.jpg"
    img = Image.open(img_path)

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