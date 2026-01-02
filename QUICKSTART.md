# 빠른 시작 가이드 (QUICKSTART)

## 1. 설치

```bash
# 프로젝트 디렉토리로 이동
cd sam_project

# 설치 스크립트 실행
bash install.sh
```

설치 중 SAM 모델 다운로드를 선택할 수 있습니다:
- **vit_h** (2.4GB): 가장 높은 정확도, RTX 4060 8GB에서 사용 가능
- **vit_l** (1.2GB): 균형잡힌 성능
- **vit_b** (375MB): 빠른 속도, 실시간 처리에 적합

## 2. 가상환경 활성화

```bash
source venv/bin/activate
```

## 3. 모델 다운로드 (설치 시 건너뛴 경우)

```bash
# vit_h 모델 다운로드 (권장)
python download_model.py --model vit_h

# 또는 다른 모델
python download_model.py --model vit_l
python download_model.py --model vit_b

# 모든 모델 다운로드
python download_model.py --model all
```

## 4. 기본 사용법

### 자동 마스크 생성

```bash
python main.py --image your_image.jpg --mode auto
```

### 포인트 프롬프트

```bash
# 특정 위치(x=100, y=100)의 객체 분할
python main.py --image your_image.jpg --mode point --points "[[100,100]]"
```

### 박스 프롬프트

```bash
# 박스 영역(x1=50, y1=50, x2=300, y2=300)의 객체 분할
python main.py --image your_image.jpg --mode box --box "[50,50,300,300]"
```

### 배치 처리

```bash
# 폴더 내 모든 이미지 처리
python batch_process.py --input_dir ./images --output_dir ./outputs
```

### 웹캠 데모

```bash
# 실시간 세그멘테이션 (vit_b 권장)
python webcam_demo.py --model vit_b
```

## 5. 결과 확인

처리된 결과는 `outputs/` 디렉토리에 저장됩니다.

## 6. GPU 메모리 최적화

RTX 4060 8GB 사용 시:
- 큰 이미지는 자동으로 1024px로 리사이즈됩니다
- 메모리 부족 시 `--max_size 512` 옵션 사용
- 실시간 처리는 `vit_b` 모델 사용 권장

```bash
# 이미지 크기 제한
python main.py --image large_image.jpg --mode auto --max_size 512

# 더 작은 모델 사용
python main.py --image your_image.jpg --mode auto --model vit_b
```

## 7. 도움말

```bash
# 메인 프로그램 옵션 확인
python main.py --help

# 배치 처리 옵션 확인
python batch_process.py --help

# 웹캠 데모 옵션 확인
python webcam_demo.py --help
```

## 8. 문제 해결

### CUDA 오류
```bash
# CUDA 설치 확인
nvidia-smi

# PyTorch CUDA 지원 확인
python -c "import torch; print(torch.cuda.is_available())"
```

### 메모리 부족
- 더 작은 모델 사용 (vit_b)
- 이미지 크기 축소 (`--max_size` 옵션)
- 배치 처리 시 한 번에 적은 수의 이미지 처리

### 패키지 오류
```bash
# 가상환경 재생성
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 추가 정보

자세한 사용법과 기능 설명은 `README.md`를 참조하세요.
