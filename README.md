# SAM 객체 분류 및 인식 프로젝트

Meta의 Segment Anything Model (SAM)을 사용한 객체 분류 및 인식 프로젝트입니다.

## 시스템 요구사항

- Ubuntu 22.04
- NVIDIA RTX 4060 8GB
- CUDA 11.8 이상
- Python 3.8 이상

## 설치 방법

### 1. CUDA 및 cuDNN 설치 (필요시)

```bash
# CUDA Toolkit 설치 확인
nvidia-smi

# CUDA가 없다면 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

### 2. Python 가상환경 생성

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate
```

### 3. 필요한 패키지 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. SAM 모델 체크포인트 다운로드

```bash
# 모델 다운로드 스크립트 실행
python download_model.py
```

사용 가능한 모델:
- `vit_h` (default, 2.4GB) - 가장 높은 정확도
- `vit_l` (1.2GB) - 균형잡힌 성능
- `vit_b` (375MB) - 빠른 속도

## 사용 방법

### 기본 실행

```bash
# 단일 이미지 처리
python main.py --image path/to/image.jpg

# 자동 마스크 생성
python main.py --image path/to/image.jpg --mode auto

# 포인트 프롬프트 사용
python main.py --image path/to/image.jpg --mode point --points "[[100,100]]"

# 박스 프롬프트 사용
python main.py --image path/to/image.jpg --mode box --box "[50,50,200,200]"
```

### 배치 처리

```bash
# 폴더 내 모든 이미지 처리
python batch_process.py --input_dir path/to/images --output_dir path/to/output
```

### 실시간 웹캠 처리

```bash
python webcam_demo.py
```

## 프로젝트 구조

```
.
├── README.md
├── requirements.txt
├── download_model.py          # 모델 다운로드 스크립트
├── main.py                     # 메인 실행 파일
├── batch_process.py            # 배치 처리 스크립트
├── webcam_demo.py              # 웹캠 데모
├── utils/
│   ├── __init__.py
│   ├── sam_predictor.py       # SAM 예측 유틸리티
│   ├── visualization.py        # 시각화 함수
│   └── image_utils.py          # 이미지 처리 유틸리티
├── models/                     # 다운로드된 모델 저장 위치
└── outputs/                    # 결과 이미지 저장 위치
```

## 주요 기능

1. **자동 마스크 생성**: 이미지 내 모든 객체 자동 감지 및 세그멘테이션
2. **포인트 프롬프트**: 클릭한 위치의 객체 세그멘테이션
3. **박스 프롬프트**: 지정한 박스 영역의 객체 세그멘테이션
4. **배치 처리**: 여러 이미지 동시 처리
5. **실시간 처리**: 웹캠을 통한 실시간 객체 세그멘테이션

## 문제 해결

### CUDA 메모리 부족
- 더 작은 모델 사용 (vit_b)
- 이미지 크기 축소

### 느린 처리 속도
- GPU 사용 확인: `nvidia-smi`로 GPU 활용 확인
- 배치 크기 조정

## 참고 자료

- [SAM 공식 GitHub](https://github.com/facebookresearch/segment-anything)
- [SAM 논문](https://arxiv.org/abs/2304.02643)
