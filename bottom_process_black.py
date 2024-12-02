import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def detect_text_in_dark_background(row_section):
    """검은 배경에서 회색 텍스트만 감지"""
    # 먼저 배경이 검은색인지 확인 (대부분의 픽셀이 어두움)
    is_dark = np.mean(row_section) < 50
    
    if not is_dark:
        return 0, None, {'valid_components': 0, 'valid_pixels': 0}
        
    # 회색 픽셀만 감지 (150~235 범위)
    gray_pixels = np.all((row_section > 150) & (row_section < 235), axis=2)
    
    # Connected Component Analysis
    binary = gray_pixels.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    valid_components = 0
    valid_gray_pixels = np.zeros_like(gray_pixels)
    
    for i in range(1, num_labels):
        component_size = stats[i, cv2.CC_STAT_AREA]
        if 20 < component_size < 500:
            component_mask = (labels == i)
            valid_gray_pixels |= component_mask
            valid_components += 1
            
    gray_ratio = np.sum(valid_gray_pixels) / valid_gray_pixels.size
    
    return gray_ratio, valid_gray_pixels, {
        'valid_components': valid_components,
        'valid_pixels': np.sum(valid_gray_pixels)
    }

def is_notice_row(gray_ratio, component_info):
    """비대면 정보 행인지 판단"""
    return (gray_ratio >= 0.004 and component_info['valid_components'] >= 8)

def analyze_bottom_rows(image):
    """시간표 이미지의 하단 3개 행만 분석하여 비대면 정보 행 찾기"""
    height, width = image.shape[:2]
    row_height = 100
    
    # 마지막 3개 행의 시작 위치 계산
    start_y = max(0, height - (3 * row_height))
    
    # 비대면 정보 행 찾기
    notice_row = None
    for i in range(3):
        row_start = start_y + (i * row_height)
        row_end = min(row_start + row_height, height)
        row_section = image[row_start:row_end, :]
        
        gray_ratio, valid_pixels, comp_info = detect_text_in_dark_background(row_section)
        
        if is_notice_row(gray_ratio, comp_info):
            notice_row = {
                'start_y': row_start,
                'end_y': row_end
            }
            break
    
    return notice_row

def process_timetable(image_path, save_path=None):
    """시간표 처리: 비대면 정보 제거"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read the image {image_path}")
        return
    
    notice_row = analyze_bottom_rows(image)
    
    # 비대면 정보 행이 발견되면 그 위까지만 자르기
    if notice_row:
        processed_image = image[:notice_row['start_y'], :]
    else:
        processed_image = image
    
    # # 결과 시각화 및 저장
    # plt.figure(figsize=(15, 8))
    
    # # 원본 이미지
    # plt.subplot(121)
    # plt.title('Original Image')
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # if notice_row:
    #     plt.axhline(y=notice_row['start_y'], color='r', linestyle='-', alpha=0.5, linewidth=2)
    # plt.axis('on')
    
    # # 처리된 이미지
    # plt.subplot(122)
    # plt.title('Processed Image')
    # plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    # plt.axis('on')
    
    # plt.tight_layout()
    
    # 결과 저장
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # 처리된 이미지 저장
        cv2.imwrite(str(save_path), processed_image)
        # 비교 이미지 저장
        #plt.savefig(str(save_path.parent / f'comparison_{save_path.name}'))
    
    #plt.close()
    
    return processed_image

def process_directory():
    """디렉토리 내의 모든 이미지 처리"""
    input_dir = Path('data_img')
    output_dir = Path('bottom_process_img')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() in valid_extensions:
            print(f'\n처리중인 파일: {img_path.name}')
            
            processed_path = output_dir / f'processed_{img_path.name}'
            process_timetable(img_path, processed_path)

if __name__ == "__main__":
    process_directory()