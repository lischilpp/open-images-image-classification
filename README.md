# OID-ImageClassification

### A collection of scripts to download data, train and evaluate an image classifier on Open Images using TensorFlow

# Features

 - create a list of all classes by image count
 - download images for custom lists of classes (using parallelization)
 - delete corrupt images
 - train a model of choice on the downloaded image dataset
 - evaluate the performance of the model (includes per-class accuracies)
 
 # Dependencies
-	Python 3.6 or higher
 - Pillow~7.0.0
 - numpy~1.18.5
 - tensorflow~2.3.1
 - tensorflow-hub~0.9.0
 - scikit-learn~0.23.2

other package versions may work too

 # Workflow
 1. Download the **Image IDs**, **Image labels**, **Boxes** and **Class Names** from https://storage.googleapis.com/openimages/web/download.html
 (Train, Validation and Test)
 2. Put them in a folder structure like this:
	![inputFolder.png](screenshots/input_folder.png)
3. Create folders named **out** and **processing**
4. Run the script **1_create_class_id_to_image_ids.py**
	*Output:*
	![script1.png](screenshots/script1.png)
5. Run the script **2_create_class_list_by_image_count.py**
	*Output:*
	![script2.png](screenshots/script2.png)
6. Choose class names to train your classifier on from **out/class_list_by_image_count** and put them into a **.csv** file inside **in/class_lists**
*Example*:
![script1.png](screenshots/class_list.png)
7. Adjust all options in **config.py** under **# image download** to your liking
8. Run the script **3_download_images.py**
	*Output:*
	![script3.png](screenshots/script3.png)
9. Run the script **4_delete_corrupt_images.py**
10. Adjust all options in **config.py** under **# model training** to your liking
11. Run the script **5_train_model.py**
	*Output:*
	![script5.png](screenshots/script5.png)
12. If you killed the previous script because it took to long,
you can extract your model from checkpoint using **6_extract_model_from_checkpoint.py**
13. Run the script **7_evaluate_model.py**
	*Output:*
	![script6.png](screenshots/script6.png)
14. DONE - now you have an Tensorflow Image classifier at **out/saved_model**