import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def detect_title_row(image_section):
    """
    배경색 기반으로 반대색 텍스트만 감지하는 함수
    채플(D) 텍스트 비율(0.006883)을 기준으로 판단
    """
    # 검은색 픽셀 감지 (임계값 150)
    black_pixels = np.all(image_section < 150, axis=2)
    text_ratio = np.sum(black_pixels) / black_pixels.size
    
    # 채플(D) 텍스트 비율보다 큰 경우를 텍스트 행으로 판단
    has_text = text_ratio > 0.006583
    
    return has_text, text_ratio, {'binary': black_pixels.astype(np.uint8) * 255}


def analyze_bottom_two_rows_and_process(image, save_path=None, filename=None, show_plots=True):
    """
    이미지의 하단부 분석 및 처리
    """
    height, width = image.shape[:2]
    row_height = 100
    
    results = []
    current_y = height
    
    print(f"\n하단 2개 행 분석 결과 - {filename}:")
    print("-" * 50)
    
    # 결과 시각화를 위한 figure 생성
    
    # plt.figure(figsize=(20, 12))
    # # 원본 이미지
    # plt.subplot(231)
    # plt.title('Original Image')
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('on')
    
    # 하단 2개 행 검사
    for i in range(3):
        if current_y - row_height < 0:
            break
            
        row_section = image[current_y - row_height:current_y, :]
        has_text, text_ratio, debug_images = detect_title_row(row_section)
        
        results.append({
            'row_number': i + 1,
            'y_start': current_y - row_height,
            'y_end': current_y,
            'text_ratio': text_ratio,
            'has_text': has_text
        })
        
        print(f"행 {i + 1}:")
        print(f"  위치: y = {current_y-row_height} ~ {current_y}")
        print(f"  텍스트 비율: {text_ratio:.6f}")
        print(f"  텍스트 존재 여부: {has_text}")
        
        # 디버깅 이미지 표시
        # plt.subplot(232 + i)
        # plt.title(f'Binary Text Detection (Row {i+1})')
        # plt.imshow(debug_images['binary'], cmap='gray')
        # plt.axis('on')
        
        current_y -= row_height
    
    # 텍스트가 있는 행 찾기
    text_rows = [row for row in results if row['has_text']]
    
    if text_rows:
        # 가장 위에 있는 텍스트 행의 시작점 찾기
        top_text_row = min(text_rows, key=lambda x: x['y_start'])
        
        # 텍스트 행들을 제외한 이미지 생성
        processed_image = image[:top_text_row['y_start'], :]
        
        # plt.subplot(234)
        # plt.title('Processed Image (Text Rows Removed)')
        # plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        # plt.axis('on')
    else:
        processed_image = image
        # plt.subplot(234)
        # plt.title('Original Image (No Text Rows Detected)')
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.axis('on')
    
    if save_path and filename:
        output_path = os.path.join(save_path, f'debug_{filename}')
        # plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"디버그 이미지 저장됨: {output_path}")
        
        if text_rows:
            processed_path = os.path.join(save_path, f'processed_{filename}')
            cv2.imwrite(processed_path, processed_image)
            print(f"처리된 이미지 저장됨: {processed_path}")
    
    # plt.close()
    return processed_image

def process_directory():
    input_dir = Path('data_img')
    output_dir = Path('bottom_process_img')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() in valid_extensions:
            print(f'\n처리중인 파일: {img_path.name}')
            
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Error: Could not read the image {img_path.name}")
                continue
                
            analyze_bottom_two_rows_and_process(
                image,
                save_path=str(output_dir),
                filename=img_path.name
            )

if __name__ == "__main__":
    process_directory()