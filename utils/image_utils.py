"""
이미지 처리 유틸리티 함수들
"""

import cv2
import numpy as np
from PIL import Image


def load_image(image_path):
    """
    이미지 로드
    
    Args:
        image_path: 이미지 파일 경로
    
    Returns:
        RGB 포맷의 numpy array
    """
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    # BGR을 RGB로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


def save_image(image, output_path):
    """
    이미지 저장
    
    Args:
        image: RGB 포맷의 numpy array
        output_path: 저장 경로
    """
    # RGB를 BGR로 변환
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)
    print(f"이미지 저장됨: {output_path}")


def resize_image(image, max_size=1024):
    """
    이미지 리사이즈 (비율 유지)
    
    Args:
        image: 입력 이미지
        max_size: 최대 크기 (가로/세로 중 큰 쪽)
    
    Returns:
        리사이즈된 이미지
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image
    
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return resized


def convert_to_grayscale(image):
    """
    이미지를 그레이스케일로 변환
    
    Args:
        image: RGB 이미지
    
    Returns:
        그레이스케일 이미지
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_mask_to_image(image, mask, background_color=(0, 0, 0)):
    """
    마스크를 이미지에 적용하여 배경 제거
    
    Args:
        image: 원본 이미지
        mask: 바이너리 마스크
        background_color: 배경 색상 (기본: 검정)
    
    Returns:
        마스크가 적용된 이미지
    """
    result = image.copy()
    result[~mask] = background_color
    
    return result


def extract_object(image, mask):
    """
    마스크를 사용하여 객체 추출 (투명 배경)
    
    Args:
        image: 원본 이미지
        mask: 바이너리 마스크
    
    Returns:
        RGBA 포맷의 이미지 (PIL Image)
    """
    # RGB 이미지에 알파 채널 추가
    rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = image
    rgba[:, :, 3] = mask.astype(np.uint8) * 255
    
    # PIL Image로 변환
    pil_image = Image.fromarray(rgba, 'RGBA')
    
    return pil_image


def crop_to_mask(image, mask, padding=10):
    """
    마스크 영역으로 이미지 크롭
    
    Args:
        image: 원본 이미지
        mask: 바이너리 마스크
        padding: 크롭 시 추가할 패딩
    
    Returns:
        크롭된 이미지
    """
    # 마스크의 경계 찾기
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        return image
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 패딩 추가
    h, w = image.shape[:2]
    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(h, y_max + padding)
    x_max = min(w, x_max + padding)
    
    # 크롭
    cropped = image[y_min:y_max, x_min:x_max]
    
    return cropped


def get_mask_bbox(mask):
    """
    마스크의 바운딩 박스 좌표 반환
    
    Args:
        mask: 바이너리 마스크
    
    Returns:
        [x_min, y_min, x_max, y_max]
    """
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        return None
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return [x_min, y_min, x_max, y_max]


def calculate_mask_area(mask):
    """
    마스크의 면적 계산
    
    Args:
        mask: 바이너리 마스크
    
    Returns:
        면적 (픽셀 수)
    """
    return np.sum(mask)


def merge_masks(masks):
    """
    여러 마스크를 하나로 병합
    
    Args:
        masks: 마스크 리스트
    
    Returns:
        병합된 마스크
    """
    if len(masks) == 0:
        return None
    
    merged = np.zeros_like(masks[0], dtype=bool)
    
    for mask in masks:
        merged = np.logical_or(merged, mask)
    
    return merged
