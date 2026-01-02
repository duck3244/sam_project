"""
시각화 유틸리티 함수들
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2


def show_mask(mask, ax, random_color=False):
    """
    마스크를 시각화
    
    Args:
        mask: 바이너리 마스크
        ax: matplotlib axis
        random_color: 랜덤 색상 사용 여부
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """
    포인트를 시각화
    
    Args:
        coords: 포인트 좌표
        labels: 포인트 레이블 (1: 전경, 0: 배경)
        ax: matplotlib axis
        marker_size: 마커 크기
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    ax.scatter(pos_points[:, 0], pos_points[:, 1], 
               color='green', marker='*', s=marker_size, 
               edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], 
               color='red', marker='*', s=marker_size, 
               edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    """
    박스를 시각화
    
    Args:
        box: 박스 좌표 [x1, y1, x2, y2]
        ax: matplotlib axis
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    rect = Rectangle((x0, y0), w, h, 
                     edgecolor='green', 
                     facecolor='none', 
                     linewidth=2)
    ax.add_patch(rect)


def visualize_masks(image, masks, scores=None, title="Segmentation Results"):
    """
    여러 마스크를 시각화
    
    Args:
        image: 원본 이미지
        masks: 마스크 배열
        scores: 각 마스크의 점수 (선택사항)
        title: 그래프 제목
    """
    num_masks = len(masks)
    
    if num_masks == 0:
        print("표시할 마스크가 없습니다.")
        return
    
    # 서브플롯 개수 계산
    cols = min(3, num_masks)
    rows = (num_masks + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    
    if num_masks == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, mask in enumerate(masks):
        ax = axes[idx]
        ax.imshow(image)
        show_mask(mask, ax, random_color=True)
        ax.axis('off')
        
        if scores is not None:
            ax.set_title(f'Mask {idx+1} (Score: {scores[idx]:.3f})')
        else:
            ax.set_title(f'Mask {idx+1}')
    
    # 빈 서브플롯 숨기기
    for idx in range(num_masks, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    return fig


def visualize_auto_masks(image, masks, max_display=10):
    """
    자동 생성된 마스크들을 시각화
    
    Args:
        image: 원본 이미지
        masks: 자동 생성된 마스크 리스트
        max_display: 최대 표시 개수
    """
    if len(masks) == 0:
        print("표시할 마스크가 없습니다.")
        return
    
    print(f"총 {len(masks)}개의 마스크 생성됨")
    
    # 안정성 점수로 정렬
    sorted_masks = sorted(masks, key=lambda x: x['predicted_iou'], reverse=True)
    
    # 전체 마스크를 하나의 이미지에 표시
    plt.figure(figsize=(15, 10))
    plt.imshow(image)
    
    # 색상 맵 생성
    for i, mask_data in enumerate(sorted_masks[:max_display]):
        mask = mask_data['segmentation']
        show_mask(mask, plt.gca(), random_color=True)
    
    plt.axis('off')
    plt.title(f'Auto-generated Masks (Displaying top {min(max_display, len(masks))})', 
              fontsize=16)
    plt.tight_layout()
    
    return plt.gcf()


def overlay_masks_on_image(image, masks, alpha=0.5):
    """
    이미지에 마스크를 오버레이
    
    Args:
        image: 원본 이미지
        masks: 마스크 리스트
        alpha: 투명도
    
    Returns:
        오버레이된 이미지
    """
    overlay = image.copy()
    
    for mask_data in masks:
        if isinstance(mask_data, dict):
            mask = mask_data['segmentation']
        else:
            mask = mask_data
        
        # 랜덤 색상 생성
        color = np.random.randint(0, 255, 3).tolist()
        
        # 마스크 영역에 색상 적용
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color
        
        # 오버레이
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
    
    return overlay


def save_visualization(fig, output_path):
    """
    시각화 결과를 파일로 저장
    
    Args:
        fig: matplotlib figure
        output_path: 저장 경로
    """
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"결과 저장됨: {output_path}")
