#!/usr/bin/env python3
"""
실시간 웹캠 세그멘테이션 데모
"""

import cv2
import argparse
import numpy as np
from utils import SAMPredictor


def draw_mask_overlay(frame, masks, alpha=0.5):
    """프레임에 마스크 오버레이"""
    overlay = frame.copy()
    
    for mask_data in masks:
        if isinstance(mask_data, dict):
            mask = mask_data['segmentation']
        else:
            mask = mask_data
        
        # 랜덤 색상
        color = np.random.randint(0, 255, 3).tolist()
        
        # 마스크 영역에 색상 적용
        overlay[mask] = color
    
    # 블렌딩
    result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='SAM 웹캠 데모')
    
    parser.add_argument(
        '--model',
        type=str,
        default='vit_b',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='SAM 모델 타입 (default: vit_b - 실시간 처리에 적합)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='모델 체크포인트 경로'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='카메라 인덱스 (default: 0)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='프레임 너비 (default: 640)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='프레임 높이 (default: 480)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=10,
        help='처리할 FPS (default: 10)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SAM 웹캠 데모")
    print("=" * 60)
    print("사용법:")
    print("  - 'q' 또는 ESC: 종료")
    print("  - 's': 현재 프레임 저장")
    print("=" * 60)
    
    # SAM Predictor 초기화
    print("모델 로딩 중...")
    predictor = SAMPredictor(
        model_type=args.model,
        checkpoint_path=args.checkpoint
    )
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    print("웹캠 시작됨. 'q' 또는 ESC를 눌러 종료하세요.")
    
    frame_count = 0
    skip_frames = max(1, 30 // args.fps)
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            # RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 일정 프레임마다 처리
            if frame_count % skip_frames == 0:
                try:
                    # 자동 마스크 생성
                    masks = predictor.generate_masks(frame_rgb)
                    
                    # 마스크 오버레이
                    if len(masks) > 0:
                        result = draw_mask_overlay(frame, masks[:20], alpha=0.4)
                        
                        # 마스크 수 표시
                        text = f"Masks: {len(masks)}"
                        cv2.putText(result, text, (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1,
                                  (0, 255, 0), 2)
                    else:
                        result = frame
                        cv2.putText(result, "No masks detected", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1,
                                  (0, 0, 255), 2)
                except Exception as e:
                    print(f"처리 오류: {e}")
                    result = frame
            
            # FPS 표시
            fps_text = f"FPS: {args.fps}"
            cv2.putText(result, fps_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 255, 0), 2)
            
            # 화면 표시
            cv2.imshow('SAM Webcam Demo', result)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # q 또는 ESC
                break
            elif key == ord('s'):  # 스크린샷
                filename = f'screenshot_{frame_count}.png'
                cv2.imwrite(filename, result)
                print(f"스크린샷 저장: {filename}")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n중단됨.")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("웹캠 종료됨.")


if __name__ == '__main__':
    main()
