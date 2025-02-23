import numpy as np
import h5py
import matplotlib.pyplot as plt
from DLHandsighns import *
import pandas as pd

def define_global_variables():
    global g_subset_num_classes
    global g_total_num_classes
    g_total_num_classes = 26
    g_subset_num_classes = 4
 
 
    
def display_image_from_row(data, row_index): 
    selected_row = data.iloc[row_index, 1:785].values  

    image = selected_row.reshape(28, 28)

    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.title(f"Label: {data.iloc[row_index, 0]}")
    plt.axis('off') 
    plt.show()

def signs_model():
    np.random.seed(1)
    hidden_layer1 = DLLayer ("Hidden 1", 64,(784,),"sigmoid","He", 1)
    #hidden_layer2 = DLLayer ("Hidden 2", 78,(64,),"sigmoid","He", 1)
    #hidden_layer3 = DLLayer ("Hidden 3", 52,(78,),"sigmoid","He", 1)
    softmax_layer = DLLayer ("Softmax 2", 26,(64,),"softmax","He", 1)
    model = DLModel()
    model.add(hidden_layer1)
    #model.add(hidden_layer2)
    #model.add(hidden_layer3)
    model.add(softmax_layer)
    model.compile("categorical_cross_entropy")
    #costs = model.train(X_train,Y_train,2000)
    #model.save_weights("C:\DL3Weights")
    #plt.plot(np.squeeze(costs))
    return model
    
def Y_to_one_hot(Y):
    global g_subset_num_classes
    global g_total_num_classes

    #num_classes = g_subset_num_classes if g_subset_num_classes != 0 else g_total_num_classes
    one_hot_Y = np.zeros((Y.size, g_total_num_classes))
    if g_subset_num_classes != 0:
        mask = np.isin(Y, np.arange(g_subset_num_classes))
        indices = np.where(mask, Y, g_subset_num_classes)
        indices = np.clip(indices, 0, g_total_num_classes - 1)
        one_hot_Y[np.arange(Y.size), indices] = 1
    else:
        one_hot_Y[np.arange(Y.size), Y] = 1
    #one_hot_Y = np.zeros((Y.size, 26))
    #one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def main():
    define_global_variables()
    global g_subset_num_classes

    curr_model = signs_model()
    curr_model.forced_num_classess = g_subset_num_classes

    #--- Prepare the data for training
    train_data = pd.read_csv(r"C:\Users\User\Downloads\Dataset_signLetters\sign_mnist_train.csv")
    #display_image_from_row(train_data, 3)
    small_train_data = train_data.head(1000)
    X_train = small_train_data.iloc[:, 1:].values.T
    X_train = X_train / 255.0 -0.5
    Y = small_train_data.iloc[:, 0].values.T
    Y_train = Y_to_one_hot(Y)
    
    #--- Train the model
    costs = curr_model.train(X_train,Y_train,1000)
    
    #--- Prepare the data for testing
    test_data = pd.read_csv(r"C:\Users\User\Downloads\Dataset_signLetters\sign_mnist_test.csv")
    small_test_data = test_data.head(1000)
    X_test = small_test_data.iloc[:, 1:].values.T
    X_test = X_test / 255.0 -0.5
    Y = small_test_data.iloc[:, 0].values.T
    Y_test = Y_to_one_hot(Y)
    print(costs)
    
    #--- Display an image
    display_image_from_row(test_data, 3)

    #--- Test the model
    print('Deep train accuracy')
    curr_model.confusion_matrix(X_train, Y_train)
    print('Deep test accuracy')
    curr_model.confusion_matrix(X_test, Y_test)

       

    
if __name__ == "__main__":
    main()