import numpy as np
import h5py
import matplotlib.pyplot as plt
from DLHandsighns import *
import pandas as pd
from PIL import Image

g_num_data_max = 10000
g_num_data1 = 10000
g_total_num_classes = 26
g_subset_num_classes = 26

####
current_directory = os.getcwd()
print("###################################################################Current Working Directory:", current_directory)
#####

g_weights_file = r"./myflaskapp/weights/Project_YA_mika"
g_load_weights_from_file =  True
g_num_px = 28


#def define_global_variables():
#    global g_subset_num_classes
#    global g_total_num_classes
#    global g_weights_file
#    global g_load_weights_from_file
#    global g_num_data

#    g_num_data = 1000
#    g_total_num_classes = 26
#    g_subset_num_classes = 4
#    g_weights_file = r"C:\Users\User\Downloads\Dataset_signLetters\Project_YA_mika"
#    g_load_weights_from_file =  False


def display_image_from_row(data, row_index, additional_label = ""):
    selected_row = data.iloc[row_index, 1:785].values

    image = selected_row.reshape(g_num_px,g_num_px)

    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.title(f"Label: {data.iloc[row_index, 0]}, Additional Label: {additional_label}")
    plt.axis('off')
    plt.show()

def create_signs_model():
    #global g_load_weights_from_file
    #global g_weights_file

    np.random.seed(1)
    if g_load_weights_from_file == False:
        hidden_layer1 = DLLayer ("l1", 200,(784,),"relu","He", 1)
        #hidden_layer2 = DLLayer ("l2", 52,(200,),"relu","He", 1)
        #hidden_layer3 = DLLayer ("l3", 52,(196,),"sigmoid","He", 1)
        #hidden_layer4 = DLLayer ("l4", 14,(52,),"sigmoid","He", 1)
        softmax_layer = DLLayer ("l5", 26,(200),"softmax","He", 1)
    else:
        hidden_layer1 = DLLayer ("l1", 200,(784,),"relu", W_initialization = f"{g_weights_file}/Layer1.h5", learning_rate = 1)
        #hidden_layer2 = DLLayer ("l2", 52,(200,),"relu",W_initialization = f"{g_weights_file}/Layer2.h5", learning_rate = 1)
        #hidden_layer3 = DLLayer ("l3", 52,(196,),"sigmoid",W_initialization = f"{g_weights_file}/Layer3.h5", learning_rate = 1)
        #hidden_layer4 = DLLayer ("l4", 14,(52,),"sigmoid",W_initialization = f"{g_weights_file}/Layer4.h5", learning_rate = 1)
        softmax_layer = DLLayer ("l5", 26,(200,),"softmax",W_initialization = f"{g_weights_file}/Layer2.h5", learning_rate = 1)

    model = DLModel()
    model.add(hidden_layer1)
    #model.add(hidden_layer2)
    #model.add(hidden_layer3)
    #model.add(hidden_layer4)
    model.add(softmax_layer)
    model.compile("categorical_cross_entropy")
    #costs = model.train(X_train,Y_train,2000)

    #plt.plot(np.squeeze(costs))
    return model

def Y_to_one_hot(Y):
    #global g_subset_num_classes
    #global g_total_num_classes

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
    #define_global_variables()
    #global g_subset_num_classes
    #global g_weights_file
    #global g_num_data

    #char = chr(ord('a') + 3)


    #g_num_data = 2000
    num_data = g_num_data1
    print(f"Num Data = {g_num_data1}")
    print(f"Num classes = {g_subset_num_classes}")

    for g_subset_num_classes_curr in range(g_subset_num_classes, g_total_num_classes+1, 4):
        print (f"Number of classes: {g_subset_num_classes_curr}")
        while num_data < g_num_data_max + 1:
            print (f"Number of data: {num_data}")
            curr_model = create_signs_model()
            curr_model.forced_num_classess = g_subset_num_classes_curr

            #--- Prepare the data for training
            train_data = pd.read_csv(r"C:\Users\noa\VSCodeProjects\Mika_Hands_Signs\dataset\sign_mnist_train.csv")
            #display_image_from_row(train_data, 3)
            small_train_data = train_data.head(num_data)
            X_train = small_train_data.iloc[:, 1:].values.T
            X_train = X_train / 255.0 -0.5
            Y = small_train_data.iloc[:, 0].values.T
            Y_train = Y_to_one_hot(Y)

            #--- Train the model
            if g_load_weights_from_file == False:
                costs = curr_model.train(X_train,Y_train,num_data)
                curr_model.save_weights(g_weights_file)
                print(costs)

            #--- Prepare the data for testing
            test_data = pd.read_csv(r"C:\Users\noa\VSCodeProjects\Mika_Hands_Signs\dataset\sign_mnist_test.csv")
            small_test_data = test_data.head(1000)
            X_test = small_test_data.iloc[:, 1:].values.T
            X_test = X_test / 255.0 -0.5
            Y = small_test_data.iloc[:, 0].values.T
            Y_test = Y_to_one_hot(Y)

            #--- Test the model
            print('Deep train accuracy')
            curr_model.confusion_matrix(X_train, Y_train)
            print('Deep test accuracy')
            curr_model.confusion_matrix(X_test, Y_test)
            num_data = num_data * 2

    """
    #--- Display an image
    image_num = 7
    my_image = X_test[:,image_num]
    my_image = my_image.reshape((784,1))
    p = curr_model.predict(my_image)
    prediction_index = np.argmax(p, axis=0)
    print (prediction_index.item())
    display_image_from_row(test_data, image_num, f"Predict: {prediction_index}")

    #--- Display ten A letters with the prediction and pictures
    curr, i = 0, 0
    while curr < Y.size and i < 10:
        if Y[curr]  == 1:
            image_num = curr
            my_image = X_test[:,image_num]
            my_image = my_image.reshape((784,1))
            p = curr_model.predict(my_image)
            prediction_index = np.argmax(p, axis=0)
            print (prediction_index.item())
            display_image_from_row(test_data, image_num, f"Predict: {prediction_index}")
            i += 1
        curr += 1
"""
    #img_path = r"C:\Users\User\Downloads\Dataset_signLetters\Project_YA_mika\Project_YA_mika\example_grayC4_test.jpg"
    '''
    img = Image.open(img_path)

    #plt.imshow(img)
    #plt.axis('off')  # Hide the axis
    #plt.show()

    img_gray = img.convert('L')

    # Save the grayscale image
    #img_gray.save("example_gray.jpg")

    # Display the grayscale image using matplotlib
    img_gray_np = np.array(img_gray)
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')  # Hide the axis
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
    print (prediction_index.item())'''
    #display_image_from_row(test_data, 0, f"Predict: {prediction_index}")

    #test_data = pd.read_csv(r"C:\Users\User\Downloads\Dataset_signLetters\sign_mnist_test.csv")
    '''small_test_data = test_data.head(1000)
    #X_test = small_test_data.iloc[:, 1:].values.T
    #my_image = X_test[:,0]
    reshapedImage = my_image.reshape((g_num_px,g_num_px))
    image1 = Image.fromarray(reshapedImage)
    image1 = image1.convert('L')
    curr = 0
    num = 1
    while curr < Y.size and num <= 10:
        if Y[curr]  == 3:
            selected_row = test_data.iloc[curr, 1:785].values

            image = selected_row.reshape(g_num_px,g_num_px,1).astype(np.uint8)
            image1 = Image.fromarray(image.squeeze(), mode='L')
            image1.save(f"example_grayD{num}_test.jpg")
            num += 1
        curr += 1'''







if __name__ == "__main__":
    main()