import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from utils.errors import ModelError
import logging

logger = logging.getLogger(__name__)

class CICIDetectionModel(nn.Module):
    def __init__(self, input_shape=(224, 224, 3)):
        super(CICIDetectionModel, self).__init__()
        self.input_shape = input_shape
        logger.info("Initializing CNN model...")
        
        # CNN Architecture for MRI analysis
        self.features = nn.Sequential(
            # First Conv Block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second Conv Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third Conv Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth Conv Block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten()
        )
        
        # Calculate flattened size
        dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1])
        dummy_output = self.features(dummy_input)
        flattened_size = dummy_output.shape[1]
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 2),  # Binary classification: CICI or No CICI
            nn.Softmax(dim=1)
        )
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        logger.info(f"Model initialized on {self.device}")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def analyze_mri(self, image):
        """Analyze MRI image and extract relevant features"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Apply preprocessing
            # 1. Normalize intensity
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            
            # 2. Apply adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(normalized.astype(np.uint8))
            
            # 3. Denoise
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # 4. Extract features
            features = {
                'mean_intensity': np.mean(denoised),
                'std_intensity': np.std(denoised),
                'entropy': self._calculate_entropy(denoised)
            }
            
            return denoised, features
            
        except Exception as e:
            logger.error(f"MRI analysis error: {str(e)}")
            raise ModelError(f"Failed to analyze MRI: {str(e)}")

    def _calculate_entropy(self, image):
        """Calculate image entropy"""
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram.ravel() / histogram.sum()
        non_zero = histogram > 0
        return -np.sum(histogram[non_zero] * np.log2(histogram[non_zero]))

    def predict(self, image):
        """Predict CICI probability from MRI image"""
        try:
            if image is None:
                raise ModelError("No image provided")

            # Convert PIL image to numpy array if needed
            if hasattr(image, 'convert'):
                image = np.array(image)

            # Analyze MRI
            processed_image, features = self.analyze_mri(image)
            
            # Prepare image for model
            processed_image = cv2.resize(processed_image, (self.input_shape[0], self.input_shape[1]))
            processed_image = processed_image.astype(np.float32) / 255.0
            
            # Convert to RGB (model expects 3 channels)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
            
            # Convert to PyTorch tensor
            tensor_image = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0)
            tensor_image = tensor_image.to(self.device)

            # Get model prediction
            self.eval()
            with torch.no_grad():
                prediction = self(tensor_image)
                
                # Log prediction details
                cici_prob = prediction[0][1].item()
                logger.info(f"CICI Probability: {cici_prob:.3f}")
                logger.info(f"Image Features: Mean={features['mean_intensity']:.2f}, "
                          f"Std={features['std_intensity']:.2f}, "
                          f"Entropy={features['entropy']:.2f}")

                return prediction.cpu().numpy()

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise ModelError(f"Prediction failed: {str(e)}")

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        try:
            # Convert data to tensors
            X_test = torch.from_numpy(X_test).float().to(self.device)
            y_test = torch.from_numpy(y_test).long().to(self.device)

            # Get predictions
            self.eval()
            with torch.no_grad():
                outputs = self(X_test)
                _, predicted = torch.max(outputs.data, 1)

            # Calculate metrics
            correct = (predicted == y_test).sum().item()
            total = y_test.size(0)
            accuracy = correct / total

            logger.info(f"Model Evaluation - Accuracy: {accuracy:.2%}")
            return accuracy, None, None  # You can add confusion matrix and class report here

        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            raise ModelError(f"Model evaluation failed: {str(e)}")

    def test_model(self):
        """Test if model is working correctly"""
        try:
            logger.info("Testing CNN model functionality...")
            
            # Create a test input tensor
            test_input = torch.randn(1, 3, self.input_shape[0], self.input_shape[1])
            test_input = test_input.to(self.device)
            
            # Try a forward pass
            self.eval()
            with torch.no_grad():
                test_output = self(test_input)
            
            # Check if output has correct shape (batch_size, num_classes)
            if test_output.shape == (1, 2):
                logger.info("Model test successful - correct output shape")
                return True
            else:
                logger.error(f"Model test failed - incorrect output shape: {test_output.shape}")
                return False
                
        except Exception as e:
            logger.error(f"Model test failed: {str(e)}")
            return False