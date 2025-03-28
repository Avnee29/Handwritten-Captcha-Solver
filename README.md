# Handwriting Captcha Solver

## About
This project is a **Handwriting Captcha Solver** that uses **Convolutional Neural Networks (CNNs)** to recognize and decode handwritten captchas automatically. It utilizes **Python, Keras, TensorFlow, and OpenCV** to preprocess images, train a deep learning model, and predict captcha characters accurately.

## Features
- **Captcha Image Processing**: Uses OpenCV to preprocess captcha images (grayscale conversion, thresholding, segmentation).
- **Deep Learning Model**: A CNN-based model trained on a dataset of handwritten captchas.
- **Prediction and Solving**: The trained model (`model.h5`) is used to predict characters from new captchas.
- **Application Interface**: A demo application to test the captcha solver.

## Project Structure
| File/Folder                 | Description |
|-----------------------------|-------------|
| `application_demo/`         | Contains test images and a demo application for solving captchas. |
| `emoji_dataset/`           | Additional dataset (if emojis are used in captchas). |
| `test_captchas/`           | Sample test captchas for evaluating the model. |
| `app.ipynb`                | Jupyter notebook for running the captcha solver application. |
| `dataset_processing.ipynb`  | Preprocesses captcha images (segmentation, filtering, etc.). |
| `character_st.txt`         | Stores character mappings for captcha labels. |
| `main.ipynb`               | Main script to run the trained model and solve captchas. |
| `model.h5`                 | Pretrained CNN model for captcha recognition. |
| `model_training.ipynb`      | Trains the CNN model using Keras and TensorFlow. |

## Installation & Requirements
### Dependencies:
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib

### Setup:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/handwriting-captcha-solver.git
   cd handwriting-captcha-solver
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model (optional, if you want to retrain):
   ```bash
   jupyter notebook
   ```
   Open `model_training.ipynb` and run the cells.
4. Run the application:
   ```bash
   jupyter notebook
   ```
   Open `app.ipynb` and follow the instructions.

## How It Works
1. **Preprocessing**: OpenCV processes captcha images (resizing, noise reduction, character segmentation).
2. **Training**: A CNN model learns character patterns from the dataset.
3. **Prediction**: The trained model predicts characters in a new captcha image.
4. **Application**: Users can input an image, and the model will return the solved captcha.

## Future Enhancements
- Improve character segmentation for better accuracy.
- Deploy as a web API for real-time captcha solving.
- Support for multi-font and colored captchas.

## Contributors
- **Your Name** (Your GitHub/Email)

## License
This project is licensed under the **MIT License**. Feel free to use and modify!

