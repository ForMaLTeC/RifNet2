# Classification of rib fracture types from postmortem computed tomography images using deep learning

## Description
Purpose: Human or time resources can sometimes fall short in medical image diagnostics, and analyzing images in full detail can be a challenging task. With recent advances in artificial intelligence, an increasing number of systems have been developed to assist clinicians in their work. In this study, the objective was to train a model that can distinguish between various fracture types on different levels of hierarchical taxonomy and detect them on 2D-image representations of volumetric postmortem computed tomography (PMCT) data.
Methods: We used a deep learning model based on the ResNet50 architecture that was pretrained on ImageNet data, and we used transfer learning to fine-tune it to our specific task. We trained our model to distinguish between “displaced”, “nondisplaced”, “ad latus”, “ad longitudinem cum contractione”, and “ad longitudinem cum distractione” fractures.
Results: Radiographs with no fractures were correctly predicted in 95%-99% of cases. Nondisplaced fractures were correctly predicted in 80%-86% of cases. Displaced fractures of the “ad latus” type were correctly- predicted in 17-18% of cases. The other two displaced types of fractures, “ad longitudinem cum contractione” and “ad longitudinem cum distractione”, were correctly predicted in 70-75%  and 64-75% of cases, respectively.
Conclusion: The model achieved the best performance when the level of hierarchical taxonomy was high, while it had more difficulties when the level of hierarchical taxonomy was lower. Overall, deep learning techniques constitute a reliable solution for forensic pathologists and medical practitioners seeking to reduce workload.

## Overview on workflow and evaluation

![Overview figure paper](Fig1.png "overview image")

**Fig. 1: Workflow of the automated rib fracture classification pipeline**
Each volumetric PMCT scan of the rib cage was transformed into a corresponding 2D representation. If the representation did not display any fracture (“no fracture”), we collected a series of sample images (each measuring 99×99 pixels) using a sliding window. Then, we randomly drew from a subset of those samples. If the representation displayed rib fractures, we collected a sample at the exact position of the fracture with an additional set of 16 samples. The additional set was obtained using data augmentation by sliding the 99×99-pixel window in each of the four cardinal directions in 10-pixel steps. The samples from the four fracture types and the “no fracture” samples were fed into a ResNet50 architecture for training and testing. We validated the performance of our model on three levels of hierarchical taxonomy: (1) a high-level task where the model distinguished between “fracture” and “no fracture”, (2) a mid-level task to assess how well the model could classify “nondisplaced” and “displaced” fractures, and (3) a low-level task to validate the performance of the model in classifying the three different types of displaced fractures “ad latus” (sideways), “ad longitudinem cum contractione” (in long axis compressed fracture) or “ad longitudinem cum distractione” (in long axis with gap between the fragments).


## Folder GUI

In the folder **GUI** we provide a user-friendly graphical user interface that allows to use the model and make predictions on PMCT images. For further details see here: https://github.com/ForMaLTeC/RifNet2/tree/master/GUI.

## Jupyter notebook Train_val_ribfractures.ipynb

**The notebook starts from after preprocessing the data and contains the following major parts:**

- *Create single dataset:* 
This part creates a single split dataset of training, (validation) and testing. 
- *Create crossvalidation dataset:*
If in *create single dataset* only training and testing data was created, this part splits the training data into k folds for crossvalidation.
- *Train model in CV to tune hyperparameters:*
Here, the model is trained with k-fold CV.
- *Validate the model:*
Validation of the CV with best accuracy or lowest loss.
- *Train full model with best hyperparameters and predict on test set:*
Final training on the full data with best hyperparameters assessed in the CV.
- *Predict on unseen data:*
Prediction on the test set and evaluate scores with aggregation and plots.

## Authors
Victor Ibañez
Akos Dobay

## Publication
https://link.springer.com/article/10.1007/s12024-023-00751-x

## Contact
akos.dobay@uzh.ch

