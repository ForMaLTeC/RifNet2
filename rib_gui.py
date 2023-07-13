# coding=utf-8

'''
RIB FRACTURE GUI
11.07.2023
Victor Ibañez
'''

# --------------------------------------------------------------------
# LIBRARY IMPORT
# --------------------------------------------------------------------


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras import optimizers
import os
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.metrics import f1_score

# --------------------------------------------------------------------
# FUNCTIONS
# --------------------------------------------------------------------

# function which is needed in pretrained network
def weighted_f1(y_true, y_pred):
    
    y_true = np.argmax(y_true.numpy(), axis=1).flatten()
    y_pred = np.argmax(y_pred.numpy(), axis=1).flatten()
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return f1

# upscale images
def upscale(img):
    img = img[250:750, 150:1150]
    scale_percent = 300 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) # resize image
    return resized

# sub-divide images into small images
def sub_divide(resized):
    
    mapping_array = np.zeros((int(resized.shape[0]/50),int(resized.shape[1]/50)))
    small_img_list = []
    coord_list = []
    shift = 50
    
    for m in range(0,resized.shape[0]-shift,shift):
        for n in range(0,resized.shape[1]-shift,shift):
            small_img = resized[m:m+99,n:n+99]
            if np.mean(small_img) < 1:
                mapping_array[int(m/50)][int(n/50)] = 0
                continue
            else:
                small_img = small_img/255.
                small_img_list.append(small_img)
                coord_list.append([m,n])
                mapping_array[int(m/50)][int(n/50)] = 1
                
    return small_img_list, coord_list, m, n, mapping_array


def predict(small_img_list, model):
    
    small_img_arr = np.array(small_img_list)
    results_prob = model.predict(small_img_arr)
    results_classes = np.argmax(results_prob,axis=1)

    return results_classes, results_prob


def transparency_func(value):
    if value > 0 and value <= 0.2:
        return 0.05
    if value > 0.2 and value <= 0.4:
        return 0.15
    if value > 0.4 and value <= 0.6:
        return 0.25
    if value > 0.6 and value <= 0.8:
        return 0.35
    if value > 0.8 and value <= 1:
        return 0.5  


def create_mask(resized, results, results_prob, coord_list):
    
    all_masks = []

    for nb in range(5):

        prob_masks = []

        for p in range(0,10,1):

            legend_nbs = [0.15,0.25,0.35,0.45,0.6]
            img_to_process = resized.copy()
            overlay = resized.copy()
            current_color_fill = color_list_fill[nb]
            rows = resized.shape[1]
            row = 0
            for r in range(len(results)):
                    
                if r != 0 and r % rows == 0:
                    row += 1
                    
                m,n = coord_list[r][0],coord_list[r][1]
                
                if results[r] == nb and results_prob[r][nb] > (p*0.1):  
                    
                    sub_img = overlay[m:m+99,n:n+99]
                    current_rect = np.ones(sub_img.shape, dtype=np.uint8)
                    current_rect[0:99,0:99] = current_color_fill
                    current_prob = results_prob[r][nb]
                    current_alpha = transparency_func(current_prob)
                    add = cv2.addWeighted(sub_img,0.5,current_rect,current_alpha,0)
                    img_to_process[m:m+99,n:n+99] = add

            # Add legend with probabilities
            y_start = 50
            quad_size = 50 
            x_start = 1500
            y_start = 40
            x_shift = 60
            y_shift = 0
            cv2.putText(img_to_process, 'probability: 0', (x_start-460, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA, False)
            cv2.putText(img_to_process, '1', (x_start+350, 85), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA, False)

            for n_iter,legend_nb in enumerate(legend_nbs):
                sub_img = overlay[y_start+(n_iter*y_shift):y_start+quad_size+(n_iter*y_shift), x_start+(n_iter*x_shift):x_start+quad_size+(n_iter*x_shift)]
                current_rect = np.ones(sub_img.shape, dtype=np.uint8)
                current_rect[0:quad_size,0:quad_size] = current_color_fill
                #current_alpha = transparency_func(legend_nb)
                add = cv2.addWeighted(sub_img,0.5,current_rect,legend_nb,0)
                img_to_process[y_start+(n_iter*y_shift):y_start+quad_size+(n_iter*y_shift), x_start+(n_iter*x_shift):x_start+quad_size+(n_iter*x_shift)] = add
                #cv2.putText(img_to_process, f'>= {n_iter/5}', ((x_start+(n_iter*x_shift))+70, (y_start+(n_iter*y_shift))+40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA, False)

            prob_masks.append(img_to_process)
        all_masks.append(prob_masks)
    
    return all_masks

def classify_image(inp):

    global all_masks
    global resized
    global results_prob

    resized = upscale(inp)
    small_img_list,coord_list,m,n,mapping_array = sub_divide(resized)
    results,results_prob = predict(small_img_list, model)
    all_masks = create_mask(resized, results, results_prob, coord_list)
    
    return 'prediction finished.'

def change_img_output_slider(choice,prob):
    prob_map = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    to_display = prob_map.index(prob)
    
    if choice == "none":
        return resized
    elif choice == "nondisplaced":
        return all_masks[0][to_display]
    elif choice == "displaced latus":
        return all_masks[4][to_display]
    elif choice == "displaced longitudinem cum contractione":
        return all_masks[2][to_display]
    elif choice == "displaced longitudinem cum distractione":
        return all_masks[1][to_display]
    else:
        return all_masks[3][to_display]


# --------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------


# define optimizer
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import Precision, Recall
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
learning_rate = 0.00015
opt = Adam(learning_rate=0.0001)
loss_type = 'categorical_crossentropy' 

model_path = '/Users/victor/Desktop/work/Institute_of_Forensic_Medicine/gui_rifnet/ResNet50_full_model.h5'

dependencies = {
    'weighted_f1': weighted_f1
}
model = tf.keras.models.load_model(model_path, custom_objects=dependencies, compile=False)
model.compile(loss=loss_type,
                  optimizer=opt,
                  metrics=[Precision(),Recall(),weighted_f1], run_eagerly=True)

color_list_fill = [(255,102,255),(102,255,102),(102,255,255),(102,102,255),(107,178,255),(255,102,102),(255,171,102)]
color_list_box = [(255,0,255),(0,255,0),(0,255,255),(0,0,255),(0,128,255),(255,0,0),(234,123,10)]
classes = ['nondisplaced','displaced_long_cum_dist','displaced_long_cum_cont','no_fracture','displaced_latus']




gui = gr.Blocks()

with gui:

    gr.Markdown(
    """
    # Rib fracture detection
    Choose an image and press "run prediction" button below:
    """)

    img_file = gr.Image(label="Initial image")
    
    b1 = gr.Button("run prediction")
    text = gr.Textbox(label="prediction outcome:")
    
    b1.click(classify_image, inputs=img_file, outputs=text)

    radio = gr.Radio(
        ["none","no fracture", "nondisplaced", "displaced latus", 
         "displaced longitudinem cum distractione", "displaced longitudinem cum contractione"], label="Classes to display:"
    )
    slider = gr.Slider(0, 0.9, value=0.5, step=0.1, label="probability")
    final_mask = gr.Image(label="output image")
    
    slider.change(fn=change_img_output_slider, inputs=[radio,slider], outputs=final_mask)
    radio.change(fn=change_img_output_slider, inputs=[radio,slider], outputs=final_mask)

gui.launch()