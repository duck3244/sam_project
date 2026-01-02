#!/usr/bin/env python3
"""
SAM 모델 체크포인트 다운로드 스크립트
"""

import os
import argparse
import requests
from tqdm import tqdm


# 모델 URL 정보
MODEL_URLS = {
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
}

MODEL_SIZES = {
    'vit_h': '2.4GB',
    'vit_l': '1.2GB',
    'vit_b': '375MB',
}


def download_file(url, destination):
    """파일 다운로드 함수"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            progress_bar.update(size)
    
    print(f"다운로드 완료: {destination}")


def main():
    parser = argparse.ArgumentParser(description='SAM 모델 체크포인트 다운로드')
    parser.add_argument(
        '--model',
        type=str,
        default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b', 'all'],
        help='다운로드할 모델 선택 (default: vit_h)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./models',
        help='모델 저장 디렉토리 (default: ./models)'
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 다운로드할 모델 결정
    if args.model == 'all':
        models_to_download = list(MODEL_URLS.keys())
    else:
        models_to_download = [args.model]
    
    # 모델 정보 출력
    print("=" * 60)
    print("SAM 모델 다운로드")
    print("=" * 60)
    for model in models_to_download:
        print(f"모델: {model} (크기: {MODEL_SIZES[model]})")
    print("=" * 60)
    print()
    
    # 모델 다운로드
    for model in models_to_download:
        url = MODEL_URLS[model]
        filename = os.path.basename(url)
        destination = os.path.join(args.output_dir, filename)
        
        # 이미 파일이 존재하는지 확인
        if os.path.exists(destination):
            print(f"모델 파일이 이미 존재합니다: {destination}")
            response = input("다시 다운로드하시겠습니까? (y/n): ")
            if response.lower() != 'y':
                print("스킵합니다.")
                continue
        
        print(f"\n다운로드 중: {model}")
        print(f"URL: {url}")
        print(f"저장 위치: {destination}")
        
        try:
            download_file(url, destination)
        except Exception as e:
            print(f"다운로드 실패: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("다운로드 완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()
