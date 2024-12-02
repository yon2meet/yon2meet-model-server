import cv2
import numpy as np
from pathlib import Path
import bottom_process_black
import bottom_process_white

def is_dark_background(image):
   """
   이미지의 상단, 중단, 하단 영역을 모두 검사하여 검은 배경인지 판단
   각 영역에서 50% 이상이 어두운 픽셀이어야 검은 배경으로 판단
   """
   height = image.shape[0]
   section_height = height // 3  # 이미지를 3등분
   
   # 상단, 중단, 하단 영역 추출
   top_section = image[0:section_height, :]
   middle_section = image[section_height:2*section_height, :]
   bottom_section = image[2*section_height:height, :]
   
   # 각 영역별 어두운 픽셀 비율 계산
   def calculate_dark_ratio(section):
       dark_pixels = np.all(section < 50, axis=2)
       return np.sum(dark_pixels) / dark_pixels.size
   
   top_ratio = calculate_dark_ratio(top_section)
   middle_ratio = calculate_dark_ratio(middle_section)
   bottom_ratio = calculate_dark_ratio(bottom_section)
   
   # 모든 영역이 50% 이상 어두울 때 검은 배경으로 판단
   return (top_ratio > 0.4 or middle_ratio > 0.4 or bottom_ratio > 0.4)

def process_directory():
    """디렉토리 내의 모든 이미지 처리"""
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
            
            # 배경색 판단
            if is_dark_background(image):
                print("검은 배경 처리 적용")
                bottom_process_black.process_timetable(img_path, output_dir / f'processed_{img_path.name}')
            else:
                print("흰색 배경 처리 적용")
                bottom_process_white.analyze_bottom_two_rows_and_process(
                    image,
                    save_path=str(output_dir),
                    filename=img_path.name
                )

if __name__ == "__main__":
    process_directory()