"""
SAM 예측 유틸리티 클래스
"""

import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


class SAMPredictor:
    """SAM 모델을 사용한 예측 클래스"""
    
    def __init__(self, model_type='vit_h', checkpoint_path=None, device=None):
        """
        Args:
            model_type: 모델 타입 ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path: 모델 체크포인트 경로
            device: 사용할 디바이스 ('cuda' 또는 'cpu')
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"사용 디바이스: {self.device}")
        
        if checkpoint_path is None:
            checkpoint_path = self._get_default_checkpoint(model_type)
        
        print(f"모델 로딩 중: {checkpoint_path}")
        
        # SAM 모델 로드
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        
        # Predictor 초기화
        self.predictor = SamPredictor(sam)
        
        # Automatic Mask Generator 초기화
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        
        print("모델 로딩 완료")
    
    def _get_default_checkpoint(self, model_type):
        """기본 체크포인트 경로 반환"""
        checkpoint_files = {
            'vit_h': 'models/sam_vit_h_4b8939.pth',
            'vit_l': 'models/sam_vit_l_0b3195.pth',
            'vit_b': 'models/sam_vit_b_01ec64.pth',
        }
        return checkpoint_files.get(model_type)
    
    def set_image(self, image):
        """이미지 설정"""
        self.predictor.set_image(image)
    
    def predict_with_points(self, point_coords, point_labels):
        """
        포인트 프롬프트를 사용한 예측
        
        Args:
            point_coords: 포인트 좌표 [[x, y], ...]
            point_labels: 포인트 레이블 [1, ...] (1: 전경, 0: 배경)
        
        Returns:
            masks: 예측된 마스크
            scores: 신뢰도 점수
            logits: 로짓
        """
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        
        return masks, scores, logits
    
    def predict_with_box(self, box):
        """
        박스 프롬프트를 사용한 예측
        
        Args:
            box: 박스 좌표 [x1, y1, x2, y2]
        
        Returns:
            masks: 예측된 마스크
            scores: 신뢰도 점수
            logits: 로짓
        """
        box = np.array(box)
        
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=False,
        )
        
        return masks, scores, logits
    
    def predict_with_points_and_box(self, point_coords, point_labels, box):
        """
        포인트와 박스 프롬프트를 함께 사용한 예측
        
        Args:
            point_coords: 포인트 좌표
            point_labels: 포인트 레이블
            box: 박스 좌표
        
        Returns:
            masks: 예측된 마스크
            scores: 신뢰도 점수
            logits: 로짓
        """
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)
        box = np.array(box)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        
        return masks, scores, logits
    
    def generate_masks(self, image):
        """
        자동 마스크 생성
        
        Args:
            image: 입력 이미지 (RGB numpy array)
        
        Returns:
            masks: 생성된 마스크 리스트
        """
        masks = self.mask_generator.generate(image)
        return masks
    
    def get_best_mask(self, masks, scores):
        """가장 높은 점수의 마스크 반환"""
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]
