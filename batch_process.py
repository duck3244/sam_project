#!/usr/bin/env python3
"""
배치 이미지 처리 스크립트
"""

import os
import argparse
from tqdm import tqdm
from pathlib import Path

from utils import (
    SAMPredictor,
    load_image,
    save_image,
    resize_image,
    visualize_auto_masks,
    save_visualization,
    overlay_masks_on_image,
)


def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """디렉토리에서 이미지 파일 목록 가져오기"""
    image_files = []
    
    for ext in extensions:
        image_files.extend(Path(directory).glob(f'*{ext}'))
        image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def process_image(predictor, image_path, output_dir, max_size=1024, save_overlay=True):
    """단일 이미지 처리"""
    try:
        # 이미지 로드
        image = load_image(str(image_path))
        
        # 리사이즈
        image = resize_image(image, max_size=max_size)
        
        # 자동 마스크 생성
        masks = predictor.generate_masks(image)
        
        # 파일명 생성
        base_name = image_path.stem
        
        # 시각화 저장
        fig = visualize_auto_masks(image, masks, max_display=20)
        viz_path = os.path.join(output_dir, f'{base_name}_masks.png')
        save_visualization(fig, viz_path)
        
        # 오버레이 저장 (옵션)
        if save_overlay:
            overlay = overlay_masks_on_image(image, masks, alpha=0.5)
            overlay_path = os.path.join(output_dir, f'{base_name}_overlay.png')
            save_image(overlay, overlay_path)
        
        return len(masks), None
        
    except Exception as e:
        return 0, str(e)


def main():
    parser = argparse.ArgumentParser(description='SAM 배치 이미지 처리')
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='입력 이미지 디렉토리'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='결과 저장 디렉토리'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='SAM 모델 타입 (default: vit_h)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='모델 체크포인트 경로'
    )
    parser.add_argument(
        '--max_size',
        type=int,
        default=1024,
        help='이미지 최대 크기 (default: 1024)'
    )
    parser.add_argument(
        '--no_overlay',
        action='store_true',
        help='오버레이 이미지 저장 안 함'
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 이미지 파일 목록
    image_files = get_image_files(args.input_dir)
    
    if len(image_files) == 0:
        print(f"이미지 파일을 찾을 수 없습니다: {args.input_dir}")
        return
    
    print("=" * 60)
    print("SAM 배치 처리")
    print("=" * 60)
    print(f"입력 디렉토리: {args.input_dir}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"이미지 개수: {len(image_files)}")
    print("=" * 60)
    
    # SAM Predictor 초기화
    predictor = SAMPredictor(
        model_type=args.model,
        checkpoint_path=args.checkpoint
    )
    
    # 처리 통계
    total_masks = 0
    failed_images = []
    
    # 배치 처리
    for image_path in tqdm(image_files, desc="Processing"):
        num_masks, error = process_image(
            predictor,
            image_path,
            args.output_dir,
            max_size=args.max_size,
            save_overlay=not args.no_overlay
        )
        
        if error:
            failed_images.append((image_path.name, error))
        else:
            total_masks += num_masks
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("처리 완료!")
    print("=" * 60)
    print(f"처리된 이미지: {len(image_files) - len(failed_images)}/{len(image_files)}")
    print(f"총 생성된 마스크: {total_masks}")
    
    if failed_images:
        print(f"\n실패한 이미지 ({len(failed_images)}개):")
        for name, error in failed_images:
            print(f"  - {name}: {error}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
