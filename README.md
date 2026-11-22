# CP467GroupProject
Group Project repository for CP467A, taught by prof. Zia Ud Din, Group members: David Hu, Jacob Jung, Roshni Mahindru, Paarth Baaga, Linda Basha
# Traffic Sign Recognition: Classical vs Deep Learning

This project compares a classical computer vision approach with a deep learning approach for traffic sign recognition using the German Traffic Sign Recognition Benchmark (GTSRB).

The goal is to evaluate not only classification accuracy, but also how robust each approach is under degraded image conditions such as reduced brightness, added noise, and motion blur.

## Approaches

### HOG + SVM
The classical pipeline resizes and converts images to grayscale, extracts Histogram of Oriented Gradients (HOG) features, and uses a Support Vector Machine (SVM) for classification.

### Convolutional Neural Network
The deep learning pipeline uses a CNN implemented in PyTorch to automatically learn visual features and classify traffic signs.

## Evaluation

Both models are evaluated using:

- Classification accuracy
- Macro F1-score
- Confusion matrices
- Training time
- Average inference time per image

Testing is performed on both the original GTSRB test set and degraded versions containing:

- Reduced brightness
- Added noise
- Motion blur

Additional preprocessing methods such as histogram equalization and Gaussian filtering may also be evaluated.

## Dataset

German Traffic Sign Recognition Benchmark (GTSRB)

- 43 traffic sign classes
- 39,209 training images
- 12,630 test images

Dataset source:  
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

## Technologies

- Python
- OpenCV
- scikit-image
- scikit-learn
- PyTorch
- torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Google Colab

## Team

- Jacob Jung
- Linda Basha
- Paarth Bagga
- David Hu
- Roshni Mahindru

## Project Goal

The final comparison will examine the trade-off between the higher expected accuracy and robustness of CNNs and the lower computational cost and greater simplicity of a HOG + SVM pipeline.
