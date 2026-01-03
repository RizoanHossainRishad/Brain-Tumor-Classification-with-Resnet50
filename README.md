# Brain Tumor Classification using ResNet50 and Deployment using BRISC2025 Dataset!

Brain tumor prediction and tumor type classification using ResNet50 from Torchvision and deploying it via FastAPI. Streamlit has been used to develop the frontend of this project to ensure a user friendly and simplified experience.


# Dataset

The dataset used for training , validation and testing is "BRISC 2025: Brain Tumor MRI Dataset for Segmentation and Classification" available on kaggle.  You can find the dataset [here](https://www.kaggle.com/datasets/briscdataset/brisc2025). **BRISC** is a high-quality, expert-annotated MRI dataset curated for **brain tumor segmentation and classification**. It addresses common limitations in existing datasets (e.g., BraTS, Figshare), including class imbalance, narrow tumor focus, and annotation inconsistencies.
This dataset includes:
-   **6,000 T1-weighted MRI images**
-   **Four classes**:  _Glioma_,  _Meningioma_,  _Pituitary Tumor_, and  _No Tumor_
-   **Pixel-wise segmentation masks**  validated by physicians and radiologists
-   **Three anatomical planes**: Axial, Coronal, and Sagittal

## Dataset Preprocessing

### Train / Validation / Test Split
The dataset is split into **training**, **validation**, and **test** sets to ensure proper model evaluation and to avoid data leakage.
-   **Training set**: 80%
-   **Validation set**: 10%
-   **Test set**: 10%
The split is performed using image file paths rather than physically creating new directories. This allows flexible dataset handling while keeping the original dataset structure intact.

### Image Transformations
All images are resized to a fixed input size compatible with **ResNet-50**.
#### Training Transformations and data augmentations
Used during training to prepare images for the model:
-   Resize images to **224 × 224** 
- Random horizontal flips ( probability of 50%) (DA)
- Random Rotation of 5 degrees (DA)
- Color Jitter (DA)
-   Convert images to PyTorch tensors
-   Normalize pixel values to **[0, 1]**
#### Validation & Test Transformations
Validation and test images use the same preprocessing as training **without any data augmentation**, ensuring fair evaluation.
### Normalization
The `ToTensor()` transformation automatically scales image pixel values from **[0, 255]** to **[0, 1]**, matching the preprocessing used during inference and ONNX deployment.

## Workflow / Methodology

 - Load Dataset
 - train-val-test split
 - data preprocessing/augmentation
 - Path mapping of train-val-test data subsets
 - Trained using ResNet50
 - Validation for tuning
 - Testing using testset
 - Model evaluation
 - Onnx Model Saving
 - FastAPI deployment
 - Streamlit Frontend development

## Required Packages
Required libraries and packages list can be found in [requirements.txt](https://github.com/RizoanHossainRishad/Brain-Tumour-Prediction-with-Resnet50/blob/main/requirements.txt) file. 
## Results

|Model|Epoch  |Accuracy |
|------------------------------------------------------------------------------------------------------------------------------------------|------|----------------------|
| ResNet50                                                                                                                                         | 8 | 0.9883|


## FastAPI Deployment
### Prerequisites
Before deploying the FastAPI backend, ensure the following Python packages are installed:

    pip install fastapi uvicorn torch torchvision onnx onnxruntime numpy pillow opencv-python tqdm scikit-learn python-multipart
  -   **fastapi** – For building the API.    
-   **uvicorn** – ASGI server to run the FastAPI app.    
-   **torch** – PyTorch for deep learning models.    
-   **torchvision** – For computer vision utilities.    
-   **onnx** – To export or work with ONNX models.    
-   **onnxruntime** – To run ONNX models.    
-   **numpy** – Array and matrix operations.    
-   **pillow** – Image processing.
-   **opencv-python** – Image and video processing.    
-   **tqdm** – Progress bars for loops.    
-   **scikit-learn** – For preprocessing or ML utilities.    
-   **python-multipart** – For handling file uploads in FastAPI.
### Steps

 1. Clone the repository or Download the repository
    `git clone https://github.com/RizoanHossainRishad/Brain-Tumour-Prediction-with-Resnet50.git`
2. Pretrained model size is too large to push in github. Follow this drive link to get the .ONNX model file and the dataset. [model](https://drive.google.com/drive/folders/1IY1OV6bxaBmJZJ0aOmKx2Iw_JkGBBemz?usp=sharing) and [dataset](https://drive.google.com/drive/folders/1zAwOyxjYu3apcr3cqWHOFZ_Z7EKCl8hM?usp=sharing)
 3. Open the terminal in the project root folder
 4. Run the FastAPI server
    `uvicorn app.main:app --reload`
    -   The `--reload` flag automatically reloads on code changes (useful in development).    
    -   Access your API at: `http://127.0.0.1:8000`
  5. **Test API endpoints**
-   Use tools like **Postman** or **curl** to test endpoints.
-   Example: `POST /predict` with an image file.
-  Or check my SWAGGER UI my simply writing  `http://127.0.0.1:8000/docs`
- On Swagger UI using browser, You'll see an interactive API — click on /predict → Try it out → enter your image in the Request Body-> Choose an image from device and you will get prediction.

## Streamlit frontend
You can View the streamlit frontend to get a smooth experience.
### Perquisites
Make sure streamlit is installed by:

    pip install streamlit
You can run the streamlit by:

    streamlit run frontend/app.py

## Additional Tips ( Not required ) 
It is good practice to use a virtual environment for isolating dependencies

    python -m venv venv
    source venv/bin/activate   # Linux / Mac
    venv\Scripts\activate      # Windows

## Video Link
Video link for the project showcase can be found here: [here](https://youtu.be/_hAjd-m_HVo)

## Contact Information

 - Rizoan Hossain Rishad
	 - Email: rizoanrishad@gmail.com

 
