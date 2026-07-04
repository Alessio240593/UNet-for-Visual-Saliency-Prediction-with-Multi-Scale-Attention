# U-Net for Visual Saliency Prediction with Multi-Scale Attention

This project focuses on **visual saliency prediction**, a task in computer vision that aims to estimate which regions of an image are most likely to attract human attention.

The model takes an RGB image as input and predicts a **saliency map**, highlighting the most visually important areas in the scene.

## Model Architecture

The proposed architecture is based on a **U-Net** backbone enhanced with **Inception modules** and **Squeeze-and-Excitation (SE) blocks**.

- **U-Net** is used to preserve spatial information through skip connections between encoder and decoder stages.
- **Inception modules** are integrated to capture multi-scale feature representations, improving the model’s ability to detect salient regions at different resolutions.
- **Squeeze-and-Excitation (SE) blocks** are used to model channel-wise dependencies by adaptively reweighting feature maps, allowing the network to emphasize more informative features and suppress less useful ones.

This combination enables the network to learn both **local details and global context**, while also improving feature recalibration through channel attention, which is crucial for accurate saliency prediction.

Overall, this hybrid design leads to sharper and more precise saliency maps.

## Running the Notebook

This project is designed to be executed directly in **Google Colab**, which is the recommended environment.

Using Colab is preferred over a local setup for several reasons:

- **Computational resources**: Colab provides free access to TPUs, which significantly speeds up training and inference compared to most local machines.
- **No local storage requirements**: the dataset is automatically downloaded at runtime, avoiding the need to store several GB of data locally.
- **Reproducibility & simplicity**: the entire pipeline can be run end-to-end in a single notebook without additional setup.

## Dataset

The dataset is automatically downloaded from Kaggle at runtime:

https://www.kaggle.com/datasets/roshan401/salicon

By default, it is stored in **Google Drive**, which is used as the working directory for data persistence.

In some cases, Kaggle may require authentication before allowing the download. For this reason, the notebook includes commented code cells to authenticate using your Kaggle username and API key if needed.

During development, the dataset was accessible without authentication, so the notebook should work out of the box in most environments.

## Repository
https://github.com/Alessio240593/UNet-for-Visual-Saliency-Prediction-with-Multi-Scale-Attention

## Fallback

If the automatic download fails:

- Uncomment the authentication section in the notebook
- Provide your Kaggle credentials as instructed

If any library issues occur:

- Reset the environment
- Restart the runtime/kernel
