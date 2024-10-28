## Data and Trained Models:
Please find the data and trained models in the following link: [Data and Trained Models](https://drive.google.com/drive/folders/1BZ6l-b9vRLwxuBWz3LjX3P2_SI2If5Rn?usp=drive_link)

## To train from scratch, follow the steps below (Please contact the authors for instructions on how to navigate the notebooks shenanigans):
1. Clone the repository
2. Download the final_data.zip from the above link
3. Upzip the final_data.zip
4. Copy the final_data folder into the model folder that you would like to train. The data **MUST** be in the same folder as the training notebooks.
5. Copy the train_indices.pkl and val_indices.pkl files from the main repo folder into the model folder that you would like to train. Otherwise, uncomment the code in the notebook to generate new indices.
5. If the notebook has individual p1 and p2 files, run them one by one. If it has a combined file, run that instead.

## To evaluate the models trained by us, follow the steps below (RECOMMENDED):
1. Clone the repository
2. In the link above, navigate to Inference Testing folder
3. Download everything in the folder (IMPORTANT)
4. Unzip the test_data.zip file. Make sure to unzip in the Inference Testing folder
5. Run the inference.ipynb notebook. If you have all the model checkpoints in the same folder as the notebook, you should be able to run the notebook without any issues.