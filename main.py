#!/usr/bin/env python3
"""
SAM 객체 분류 및 인식 메인 실행 파일
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    SAMPredictor,
    load_image,
    save_image,
    resize_image,
    visualize_masks,
    visualize_auto_masks,
    show_points,
    show_box,
    save_visualization,
    apply_mask_to_image,
    extract_object,
)


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='SAM 객체 분류 및 인식')
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='입력 이미지 경로'
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
        '--mode',
        type=str,
        default='auto',
        choices=['auto', 'point', 'box', 'point_box'],
        help='세그멘테이션 모드 (default: auto)'
    )
    parser.add_argument(
        '--points',
        type=str,
        default=None,
        help='포인트 좌표 (예: "[[100,100],[200,200]]")'
    )
    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        help='포인트 레이블 (예: "[1,1]", 1: 전경, 0: 배경)'
    )
    parser.add_argument(
        '--box',
        type=str,
        default=None,
        help='박스 좌표 (예: "[50,50,200,200]")'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='결과 저장 디렉토리 (default: ./outputs)'
    )
    parser.add_argument(
        '--save_mask',
        action='store_true',
        help='마스크를 별도 이미지로 저장'
    )
    parser.add_argument(
        '--extract_object',
        action='store_true',
        help='객체를 추출하여 투명 배경으로 저장'
    )
    parser.add_argument(
        '--max_size',
        type=int,
        default=1024,
        help='이미지 최대 크기 (default: 1024)'
    )
    
    return parser.parse_args()


def process_auto_mode(predictor, image, args):
    """자동 마스크 생성 모드"""
    print("자동 마스크 생성 중...")
    
    masks = predictor.generate_masks(image)
    
    print(f"{len(masks)}개의 마스크가 생성되었습니다.")
    
    # 시각화
    fig = visualize_auto_masks(image, masks, max_display=20)
    
    # 저장
    output_path = os.path.join(args.output_dir, 'auto_masks.png')
    save_visualization(fig, output_path)
    
    # 개별 마스크 저장 (옵션)
    if args.save_mask:
        for i, mask_data in enumerate(masks[:10]):  # 상위 10개만 저장
            mask = mask_data['segmentation']
            masked_image = apply_mask_to_image(image, mask)
            mask_path = os.path.join(args.output_dir, f'mask_{i}.png')
            save_image(masked_image, mask_path)


def process_point_mode(predictor, image, args):
    """포인트 프롬프트 모드"""
    if args.points is None:
        raise ValueError("포인트 좌표를 지정해야 합니다. (--points)")
    
    # 포인트 파싱
    import ast
    point_coords = np.array(ast.literal_eval(args.points))
    
    if args.labels is None:
        point_labels = np.ones(len(point_coords))
    else:
        point_labels = np.array(ast.literal_eval(args.labels))
    
    print(f"포인트 프롬프트 사용: {len(point_coords)}개 포인트")
    
    # 이미지 설정
    predictor.set_image(image)
    
    # 예측
    masks, scores, logits = predictor.predict_with_points(point_coords, point_labels)
    
    # 가장 좋은 마스크 선택
    best_mask, best_score = predictor.get_best_mask(masks, scores)
    
    print(f"최고 점수: {best_score:.3f}")
    
    # 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(point_coords, point_labels, plt.gca())
    
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.imshow(mask, alpha=0.5, cmap='jet')
        show_points(point_coords, point_labels, plt.gca())
        plt.title(f'Mask {i+1}, Score: {score:.3f}')
        plt.axis('off')
        
        output_path = os.path.join(args.output_dir, f'point_mask_{i}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"저장됨: {output_path}")
    
    # 객체 추출 (옵션)
    if args.extract_object:
        extracted = extract_object(image, best_mask)
        extract_path = os.path.join(args.output_dir, 'extracted_object.png')
        extracted.save(extract_path)
        print(f"객체 추출됨: {extract_path}")


def process_box_mode(predictor, image, args):
    """박스 프롬프트 모드"""
    if args.box is None:
        raise ValueError("박스 좌표를 지정해야 합니다. (--box)")
    
    # 박스 파싱
    import ast
    box = np.array(ast.literal_eval(args.box))
    
    print(f"박스 프롬프트 사용: {box}")
    
    # 이미지 설정
    predictor.set_image(image)
    
    # 예측
    masks, scores, logits = predictor.predict_with_box(box)
    
    print(f"점수: {scores[0]:.3f}")
    
    # 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(masks[0], alpha=0.5, cmap='jet')
    show_box(box, plt.gca())
    plt.title(f'Box Segmentation, Score: {scores[0]:.3f}')
    plt.axis('off')
    
    output_path = os.path.join(args.output_dir, 'box_mask.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"저장됨: {output_path}")
    
    # 객체 추출 (옵션)
    if args.extract_object:
        extracted = extract_object(image, masks[0])
        extract_path = os.path.join(args.output_dir, 'extracted_object.png')
        extracted.save(extract_path)
        print(f"객체 추출됨: {extract_path}")


def process_point_box_mode(predictor, image, args):
    """포인트와 박스 프롬프트 결합 모드"""
    if args.points is None or args.box is None:
        raise ValueError("포인트와 박스 좌표를 모두 지정해야 합니다.")
    
    # 파싱
    import ast
    point_coords = np.array(ast.literal_eval(args.points))
    box = np.array(ast.literal_eval(args.box))
    
    if args.labels is None:
        point_labels = np.ones(len(point_coords))
    else:
        point_labels = np.array(ast.literal_eval(args.labels))
    
    print(f"포인트+박스 프롬프트 사용")
    
    # 이미지 설정
    predictor.set_image(image)
    
    # 예측
    masks, scores, logits = predictor.predict_with_points_and_box(
        point_coords, point_labels, box
    )
    
    # 가장 좋은 마스크
    best_mask, best_score = predictor.get_best_mask(masks, scores)
    
    print(f"최고 점수: {best_score:.3f}")
    
    # 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(best_mask, alpha=0.5, cmap='jet')
    show_points(point_coords, point_labels, plt.gca())
    show_box(box, plt.gca())
    plt.title(f'Combined Segmentation, Score: {best_score:.3f}')
    plt.axis('off')
    
    output_path = os.path.join(args.output_dir, 'combined_mask.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"저장됨: {output_path}")


def main():
    """메인 함수"""
    args = parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SAM 객체 분류 및 인식")
    print("=" * 60)
    
    # 이미지 로드
    print(f"이미지 로딩: {args.image}")
    image = load_image(args.image)
    
    # 이미지 리사이즈 (필요시)
    original_size = image.shape[:2]
    image = resize_image(image, max_size=args.max_size)
    resized_size = image.shape[:2]
    
    if original_size != resized_size:
        print(f"이미지 리사이즈: {original_size} -> {resized_size}")
    
    # SAM Predictor 초기화
    predictor = SAMPredictor(
        model_type=args.model,
        checkpoint_path=args.checkpoint
    )
    
    # 모드별 처리
    if args.mode == 'auto':
        process_auto_mode(predictor, image, args)
    elif args.mode == 'point':
        process_point_mode(predictor, image, args)
    elif args.mode == 'box':
        process_box_mode(predictor, image, args)
    elif args.mode == 'point_box':
        process_point_box_mode(predictor, image, args)
    
    print("=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
