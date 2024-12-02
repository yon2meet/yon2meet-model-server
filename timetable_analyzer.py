import cv2
import numpy as np
from collections import Counter
from pathlib import Path
import bottom_process_black
import bottom_process_white
from time_detect_test import NumberClassifier
def get_most_common_value(distances, tolerance=3):
    """
    거리 리스트에서 오차 범위 내의 값들을 그룹화하고 가장 큰 그룹의 평균 반환
    
    Parameters:
    distances (list): 거리 값들의 리스트
    tolerance (int): 오차 허용 범위 (픽셀)
    """
    if not distances:
        return None
    
    # 모든 값들을 독립된 그룹으로 시작
    groups = [[d] for d in distances]
    
    # 비슷한 값들을 그룹화
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                # 각 그룹의 평균값 계산
                avg_i = sum(groups[i]) / len(groups[i])
                avg_j = sum(groups[j]) / len(groups[j])
                
                # 두 그룹의 평균값 차이가 tolerance 이내면 병합
                if abs(avg_i - avg_j) <= tolerance:
                    groups[i].extend(groups[j])
                    groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    
    # 그룹들 중에서 가장 큰 그룹 찾기
    if not groups:
        return None
        
    largest_group = max(groups, key=len)
    
    # 가장 큰 그룹의 크기가 1이면 적절한 값을 찾지 못한 것
    if len(largest_group) == 1:
        return None
    
    # 해당 그룹의 평균값 반환
    return sum(largest_group) / len(largest_group)



def is_dark_background(image):
    """이미지가 어두운 배경인지 확인"""
    height = image.shape[0]
    section_height = height // 3
    
    def calculate_dark_ratio(section):
        dark_pixels = np.all(section < 50, axis=2)
        return np.sum(dark_pixels) / dark_pixels.size
    
    # 상단, 중단, 하단 영역 중 하나라도 어두우면 어두운 배경으로 판단
    top_section = image[0:section_height, :]
    middle_section = image[section_height:2*section_height, :]
    bottom_section = image[2*section_height:height, :]
    
    top_ratio = calculate_dark_ratio(top_section)
    middle_ratio = calculate_dark_ratio(middle_section)
    bottom_ratio = calculate_dark_ratio(bottom_section)
    
    return (top_ratio > 0.4 or middle_ratio > 0.4 or bottom_ratio > 0.4)

def preprocess_timetable(image_path):
    """시간표 이미지 전처리 - 비대면 강의 정보 제거"""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read the image {image_path}")
        return None
        
    # 배경색 확인 및 적절한 처리 방법 선택
    if is_dark_background(image):
        print("검은 배경 처리 적용")
        processed_image = bottom_process_black.process_timetable(image_path, None)
    else:
        print("흰색 배경 처리 적용")
        processed_image = bottom_process_white.analyze_bottom_two_rows_and_process(
            image,
            save_path=None,
            filename=Path(image_path).name
        )
    
    if processed_image is None:
        return image
        
    return processed_image

def detect_and_process_grid(image):
    """Grid line 검출 및 필터링"""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)
    
    min_line_length = min(height, width) // 4
    lines = cv2.HoughLinesP(edges, 
                           rho=1,
                           theta=np.pi/180,
                           threshold=50,
                           minLineLength=min_line_length,
                           maxLineGap=30)
    
    if lines is None:
        return None, None
    
    vertical_lines = []
    horizontal_lines = []
    min_length = 0.3 * min(height, width)
    
    # 각도와 길이 조건으로 필터링
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if length >= min_length:
            if angle < 10 or angle > 170:  # 수평선
                horizontal_lines.append([x1, y1, x2, y2])
            elif 80 < angle < 100:  # 수직선
                vertical_lines.append([x1, y1, x2, y2])
    
    # 간격 조건으로 필터링
    filtered_vertical = []
    filtered_horizontal = []
    
    # 수직선 간격 체크 (100~230px)
    sorted_v_lines = sorted(vertical_lines, key=lambda x: x[0])
    for i in range(len(sorted_v_lines)-1):
        current = sorted_v_lines[i]
        next_line = sorted_v_lines[i+1]
        distance = abs(next_line[0] - current[0])
        if 100 <= distance <= 230:
            if current not in filtered_vertical:
                filtered_vertical.append(current)
            if next_line not in filtered_vertical:
                filtered_vertical.append(next_line)
    
    # 수평선 간격 체크 (70~210px)
    sorted_h_lines = sorted(horizontal_lines, key=lambda x: x[1])
    for i in range(len(sorted_h_lines)-1):
        current = sorted_h_lines[i]
        next_line = sorted_h_lines[i+1]
        distance = abs(next_line[1] - current[1])
        if 70 <= distance <= 210:
            if current not in filtered_horizontal:
                filtered_horizontal.append(current)
            if next_line not in filtered_horizontal:
                filtered_horizontal.append(next_line)
    
    return filtered_vertical, filtered_horizontal

def calculate_line_distances(lines, is_vertical=True):
    """선들 사이의 거리 계산 (70px 이상만 포함)"""
    if not lines or len(lines) < 2:
        return []
    
    if is_vertical:
        sorted_lines = sorted(lines, key=lambda line: line[0])
        distances = [d for d in [abs(sorted_lines[i+1][0] - sorted_lines[i][0]) 
                    for i in range(len(sorted_lines)-1)] if d >= 70]
    else:
        sorted_lines = sorted(lines, key=lambda line: line[1])
        distances = [d for d in [abs(sorted_lines[i+1][1] - sorted_lines[i][1]) 
                    for i in range(len(sorted_lines)-1)] if d >= 70]
    
    return distances

def find_offsets(vertical_lines, horizontal_lines):
    """Grid line들로부터 적절한 offset 찾기"""
    # 첫 번째 vertical line의 x좌표 찾기
    x_coordinates = [line[0] for line in vertical_lines]
    min_x = min(x_coordinates)
    
    # 첫 번째 horizontal line의 y좌표 찾기
    y_coordinates = [line[1] for line in horizontal_lines]
    min_y = min(y_coordinates)
    
    # x_offset 결정 (50~80px 범위 확인)
    if 50 <= min_x <= 80:
        x_offset = min_x
    else:
        x_offset = 65  # 기본값
        
    # y_offset 결정 (40~60px 범위 확인)
    if 38 <= min_y <= 60:
        y_offset = min_y
    else:
        y_offset = 54  # 기본값
        
    return x_offset, y_offset

def detect_cells(image):
    """배경색에 따른 cell 검출 로직"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    is_dark_background = np.mean(gray) < 128

    if is_dark_background:
        # 어두운 배경일 때 밝은 부분을 감지 (threshold 상향)
        _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    else:
        # 밝은 배경일 때는 RGB 채널 간의 차이로 색상 검출
        rgb_max = np.max(image, axis=2)
        rgb_min = np.min(image, axis=2)
        rgb_diff = rgb_max - rgb_min
        
        binary = np.zeros_like(gray)
        
        # RGB 채널 간 차이가 threshold 이상이고
        # 전체적으로 너무 밝지 않은(흰색이 아닌) 픽셀 검출
        color_threshold = 40
        white_threshold = 230
        
        binary[(rgb_diff > color_threshold) & 
              (rgb_min < white_threshold)] = 255
        
        # 노이즈 제거
        kernel = np.ones((5,5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Connected Component Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    
    # 결과 마스크 초기화
    color_mask = np.zeros_like(binary)
    
    # 최소 cell 크기 계산
    height, width = image.shape[:2]
    min_cell_area = (width * height) * 0.005
    
    # 각 컴포넌트 분석
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        if (area >= min_cell_area and
            w/h >= 0.2 and
            h/w >= 0.2):
            
            component_mask = (labels == i).astype(np.uint8) * 255
            color_mask |= component_mask
    
    return color_mask, is_dark_background

def get_day_index(x, x_offset, cell_width):
    """x 좌표에 해당하는 요일 인덱스 계산"""
    column_positions = [x_offset + i * cell_width for i in range(7)]  # 각 열의 시작 위치
    
    for i, pos in enumerate(column_positions[:-1]):
        if pos <= x < (pos + cell_width):
            return i
    return None

def get_most_common_value(distances, tolerance=3):
    """
    거리 리스트에서 오차 범위 내의 값들을 그룹화하고 가장 큰 그룹의 평균 반환
    
    Parameters:
    distances (list): 거리 값들의 리스트
    tolerance (int): 오차 허용 범위 (픽셀)
    """
    if not distances:
        return None
    
    # 모든 값들을 독립된 그룹으로 시작
    groups = [[d] for d in distances]
    
    # 비슷한 값들을 그룹화
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                # 각 그룹의 평균값 계산
                avg_i = sum(groups[i]) / len(groups[i])
                avg_j = sum(groups[j]) / len(groups[j])
                
                # 두 그룹의 평균값 차이가 tolerance 이내면 병합
                if abs(avg_i - avg_j) <= tolerance:
                    groups[i].extend(groups[j])
                    groups.pop(j)
                    merged = True
                    break
            if merged:
                break
    
    # 그룹들 중에서 가장 큰 그룹 찾기
    if not groups:
        return None
        
    largest_group = max(groups, key=len)
    
    # 가장 큰 그룹의 크기가 1이면 적절한 값을 찾지 못한 것
    if len(largest_group) == 1:
        return None
    
    # 해당 그룹의 평균값 반환
    return sum(largest_group) / len(largest_group)


def analyze_timetable(image_path):
    """시간표 분석 전체 파이프라인"""
    print(f"Analyzing image: {image_path}")
    
    # 1. 이미지 전처리
    processed_image = preprocess_timetable(image_path)
    if processed_image is None:
        print("Failed to process image")
        return None
        
    # 2. Grid line 검출 및 필터링
    vertical_lines, horizontal_lines = detect_and_process_grid(processed_image)
    if not vertical_lines or not horizontal_lines:
        print("Grid lines not detected")
        return None
    
    # 3. Offset 계산
    x_offset, y_offset = find_offsets(vertical_lines, horizontal_lines)
    print(f"\nDetected offsets: x_offset={x_offset:.1f}, y_offset={y_offset:.1f}")
    # 4. 거리 계산
    v_distances = calculate_line_distances(vertical_lines, is_vertical=True)
    h_distances = calculate_line_distances(horizontal_lines, is_vertical=False)
    print("\nVertical distances:", v_distances)
    print("Horizontal distances:", h_distances)

    # 4. 시작 시간 인식
    classifier = NumberClassifier('best_efficientnet.pt')
    start_time, confidence, _ = classifier.classify_number(image_path, x_offset, y_offset)
    if start_time is None or start_time ==0:
        print("Failed to detect start time, using default (9)")
        start_time = 9
    else:
        print(f"Detected start time: {start_time}:00 (confidence: {confidence:.2f})")
    
    # 5. 거리 계산 및 Cell 크기 결정
    v_distances = calculate_line_distances(vertical_lines, is_vertical=True)
    h_distances = calculate_line_distances(horizontal_lines, is_vertical=False)
    
    cell_width = get_most_common_value(v_distances)
    cell_height = get_most_common_value(h_distances)
    
    if not cell_width or not cell_height:
        print("Could not determine cell size")
        return None
    print(f"\nDetected cell size: {cell_width:.1f} x {cell_height:.1f}")

    # 6. Colored cell 검출
    mask, is_dark_background = detect_cells(processed_image)
    
    # 7. 시간표 정보 추출
    height, width = processed_image.shape[:2]
    schedule = []
    days = ['월', '화', '수', '목', '금', '토', '일']
    
    min_cell_fill = 0.75
    
    
    for y in range(int(y_offset), height - int(y_offset), int(cell_height)):
        for x in range(int(x_offset), width - int(x_offset), int(cell_width)):
            cell_region = mask[y:y+int(cell_height), x:x+int(cell_width)]
            cell_fill_ratio = np.sum(cell_region == 255) / (cell_height * cell_width)
            
            if cell_fill_ratio > min_cell_fill:
                col_index = int(x / cell_width)  # offset 없이 직접 계산
                row_index = int(y / cell_height)
    
                total_columns = int(width / cell_width)

                if 0 <= col_index < total_columns:
                    day = days[col_index]
                    time = f"{start_time + row_index:02d}:00"
                    
                    schedule.append({
                        'day': day,
                        'time': time,
                        'x': x,
                        'y': y,
                        'fill_ratio': cell_fill_ratio
                    })
    
    return schedule


if __name__ == "__main__":
    # 단일 이미지 처리
    image_path = "/Users/ijunseong/Desktop/Yon2meet/dataset/data_img/KakaoTalk_Photo_2024-11-20-17-35-02 005.jpeg"
    schedule = analyze_timetable(image_path)
    
    if schedule:
        print("\n수업 시간표:")
        for class_info in sorted(schedule, key=lambda x: (x['day'], x['time'])):
            print(f"{class_info['day']}요일 {class_info['time']} 시작 (connfidence: {class_info['fill_ratio']:.2%})")
    else:
        print("Failed to analyze timetable")