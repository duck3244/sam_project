#!/bin/bash

# 프로젝트 구조 출력

echo "SAM 프로젝트 구조:"
echo ""
tree -L 2 -I 'venv|__pycache__|*.pyc|models|outputs' .
echo ""
echo "주요 파일 설명:"
echo "  - README.md: 프로젝트 문서"
echo "  - requirements.txt: Python 패키지 의존성"
echo "  - install.sh: 설치 스크립트"
echo "  - download_model.py: SAM 모델 다운로드"
echo "  - main.py: 메인 실행 파일"
echo "  - batch_process.py: 배치 처리"
echo "  - webcam_demo.py: 웹캠 데모"
echo "  - run_examples.sh: 실행 예제"
echo "  - utils/: 유틸리티 모듈"
echo "  - models/: 다운로드된 모델 저장 (생성됨)"
echo "  - outputs/: 결과 이미지 저장 (생성됨)"
