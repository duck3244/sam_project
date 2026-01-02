#!/bin/bash

# SAM 프로젝트 설치 스크립트
# Ubuntu 22.04 + RTX 4060 8GB 환경

set -e

echo "======================================================================"
echo "SAM 프로젝트 설치 시작"
echo "======================================================================"

# Python 버전 확인
echo "Python 버전 확인 중..."
python3 --version

# CUDA 확인
echo "CUDA 설치 확인 중..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "경고: nvidia-smi를 찾을 수 없습니다. CUDA가 설치되어 있는지 확인하세요."
fi

# 가상환경 생성
echo ""
echo "Python 가상환경 생성 중..."
python3 -m venv venv

# 가상환경 활성화
echo "가상환경 활성화 중..."
source venv/bin/activate

# pip 업그레이드
echo ""
echo "pip 업그레이드 중..."
pip install --upgrade pip

# PyTorch 설치 (CUDA 11.8)
echo ""
echo "PyTorch 설치 중 (CUDA 11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 나머지 패키지 설치
echo ""
echo "필요한 패키지 설치 중..."
pip install -r requirements.txt

# 디렉토리 생성
echo ""
echo "디렉토리 생성 중..."
mkdir -p models
mkdir -p outputs

# 모델 다운로드 옵션
echo ""
echo "======================================================================"
echo "SAM 모델을 다운로드하시겠습니까?"
echo "1) vit_h (2.4GB, 가장 높은 정확도)"
echo "2) vit_l (1.2GB, 균형잡힌 성능)"
echo "3) vit_b (375MB, 빠른 속도)"
echo "4) 모두 다운로드"
echo "5) 나중에 다운로드"
echo "======================================================================"
read -p "선택 (1-5): " model_choice

case $model_choice in
    1)
        echo "vit_h 모델 다운로드 중..."
        python download_model.py --model vit_h
        ;;
    2)
        echo "vit_l 모델 다운로드 중..."
        python download_model.py --model vit_l
        ;;
    3)
        echo "vit_b 모델 다운로드 중..."
        python download_model.py --model vit_b
        ;;
    4)
        echo "모든 모델 다운로드 중..."
        python download_model.py --model all
        ;;
    5)
        echo "모델 다운로드를 건너뜁니다."
        echo "나중에 'python download_model.py --model <모델명>' 으로 다운로드하세요."
        ;;
    *)
        echo "잘못된 선택입니다. 모델 다운로드를 건너뜁니다."
        ;;
esac

echo ""
echo "======================================================================"
echo "설치 완료!"
echo "======================================================================"
echo ""
echo "사용 방법:"
echo "1. 가상환경 활성화: source venv/bin/activate"
echo "2. 프로그램 실행: python main.py --image path/to/image.jpg"
echo ""
echo "자세한 사용법은 README.md를 참조하세요."
echo "======================================================================"
