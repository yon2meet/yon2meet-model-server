import torch
import torchvision
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
from collections import defaultdict
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class SimplifiedEfficientNet(nn.Module):
    def __init__(self):
        super(SimplifiedEfficientNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            nn.Conv2d(64, 384, kernel_size=1),
            nn.BatchNorm2d(384),
            nn.SiLU(),
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1, groups=384),
            nn.BatchNorm2d(384),
            nn.SiLU(),
            nn.Conv2d(384, 96, kernel_size=1),
            nn.BatchNorm2d(96),
            
            nn.Conv2d(96, 576, kernel_size=1),
            nn.BatchNorm2d(576),
            nn.SiLU(),
            nn.Conv2d(576, 576, kernel_size=3, stride=1, padding=1, groups=576),
            nn.BatchNorm2d(576),
            nn.SiLU(),
            nn.Conv2d(576, 144, kernel_size=1),
            nn.BatchNorm2d(144),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Linear(144, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class NumberClassifier:
    def __init__(self, model_path='best_efficientnet.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = SimplifiedEfficientNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def preprocess_image(self, image, x_offset, y_offset):
        print("Original image shape:", image.shape)
        
        # offset 기반으로 상단 영역 크롭
        top_region = image[:int(y_offset)+40, :int(x_offset)+5]
        print("Cropped region shape:", top_region.shape)
        
        # 크기 확대
        scale = 4.0
        enlarged = cv2.resize(top_region, None, fx=scale, fy=scale)
        print("Enlarged image shape:", enlarged.shape)
        
        # 배경이 검은색인지 확인
        is_dark = np.mean(enlarged) < 150
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        
        if is_dark:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(2,2))
            enhanced = clahe.apply(gray)
            
            gamma = 0.7
            lookUpTable = np.empty((1,256), np.uint8)
            for i in range(256):
                lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
            gamma_corrected = cv2.LUT(enhanced, lookUpTable)
            
            _, binary = cv2.threshold(gamma_corrected, 140, 255, cv2.THRESH_BINARY)
        else:
            inverted = cv2.bitwise_not(gray)
            blurred = cv2.GaussianBlur(inverted, (3,3), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((2,2), np.uint8)
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        processed = cv2.dilate(denoised, kernel, iterations=1)

        # 숫자 중앙 정렬
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            digit = processed[y:y+h, x:x+w]
            
            target_size = max(w, h) + 20
            square_img = np.zeros((target_size, target_size), dtype=np.uint8)
            x_offset = (target_size - w) // 2
            y_offset = (target_size - h) // 2
            square_img[y_offset:y_offset+h, x_offset:x_offset+w] = digit
            
            processed = square_img
        
        return processed, top_region

    def classify_number(self, image_path, x_offset, y_offset):
        print(f"\nProcessing image: {image_path}")
        original_img = cv2.imread(image_path)
        if original_img is None:
            print("Error: Could not load image")
            return None, None, None
        
        processed_img, original_crop = self.preprocess_image(original_img, x_offset, y_offset)
        pil_image = Image.fromarray(processed_img)
        
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            number = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][number].item()
            
            print(f"Predicted number: {number}")
            print(f"Confidence: {confidence:.4f}")
        
        result_img = processed_img.copy()
        cv2.putText(result_img, f"Number: {number}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 0, 0), 2)
        
        processed_resized = cv2.resize(processed_img, (100, 100))
        result_resized = cv2.resize(result_img, (100, 100))
        
        return number, confidence, {
            'original': original_crop,
            'processed': processed_resized,
            'result': result_resized
        }

def save_visualization(original, processed, result, detected_number, confidence, save_path='classification_result.jpg'):
    target_size = (100, 100)
    original_resized = cv2.resize(original, target_size)
    processed_resized = cv2.resize(processed, target_size)
    result_resized = cv2.resize(result, target_size)
    
    if len(original_resized.shape) == 2:
        original_resized = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2BGR)
    if len(processed_resized.shape) == 2:
        processed_resized = cv2.cvtColor(processed_resized, cv2.COLOR_GRAY2BGR)
    if len(result_resized.shape) == 2:
        result_resized = cv2.cvtColor(result_resized, cv2.COLOR_GRAY2BGR)
    
    combined = np.hstack((original_resized, processed_resized, result_resized))
    
    padding = 50
    combined_with_title = np.ones((combined.shape[0] + padding, combined.shape[1], 3), dtype=np.uint8) * 255
    combined_with_title[padding:, :] = combined
    
    titles = ['Original', 'Processed', f'Classified: {detected_number} ({confidence:.2f})']
    y_pos = 30
    for i, title in enumerate(titles):
        x_pos = i * 100 + 10
        cv2.putText(combined_with_title, title, (x_pos, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imwrite(save_path, combined_with_title)
    print(f"Result saved as {save_path}")

def main():
    image_path = '/Users/ijunseong/Desktop/Yon2meet/dataset/data_img/KakaoTalk_Photo_2024-11-20-16-31-49.jpeg'  # 실제 이미지 경로로 변경 필요
    
    classifier = NumberClassifier('best_efficientnet.pt')
    number, confidence, images = classifier.classify_number(image_path)
    
    if number is not None:
        print(f"Final classification - Number: {number}, Confidence: {confidence:.2f}")
        save_visualization(
            images['original'],
            images['processed'],
            images['result'],
            number,
            confidence,
            'classification_result.jpg'
        )
    else:
        print("No number classified")

if __name__ == "__main__":
    main()