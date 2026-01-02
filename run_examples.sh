#!/bin/bash

# SAM 프로젝트 실행 예제

echo "======================================================================"
echo "SAM 실행 예제"
echo "======================================================================"
echo ""

# 가상환경 활성화 확인
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "가상환경이 활성화되지 않았습니다."
    echo "다음 명령어로 활성화하세요: source venv/bin/activate"
    exit 1
fi

echo "다음 예제 중 하나를 선택하세요:"
echo ""
echo "1. 자동 마스크 생성 (이미지 내 모든 객체 자동 감지)"
echo "   python main.py --image sample.jpg --mode auto"
echo ""
echo "2. 포인트 프롬프트 (특정 위치의 객체 분할)"
echo "   python main.py --image sample.jpg --mode point --points \"[[100,100],[200,200]]\""
echo ""
echo "3. 박스 프롬프트 (박스 영역의 객체 분할)"
echo "   python main.py --image sample.jpg --mode box --box \"[50,50,300,300]\""
echo ""
echo "4. 배치 처리 (여러 이미지 동시 처리)"
echo "   python batch_process.py --input_dir ./images --output_dir ./outputs"
echo ""
echo "5. 웹캠 데모 (실시간 세그멘테이션)"
echo "   python webcam_demo.py --model vit_b"
echo ""
echo "======================================================================"
echo ""

read -p "실행할 예제 번호를 선택하세요 (1-5), 또는 Enter를 눌러 종료: " example_choice

case $example_choice in
    1)
        read -p "이미지 경로를 입력하세요: " img_path
        python main.py --image "$img_path" --mode auto
        ;;
    2)
        read -p "이미지 경로를 입력하세요: " img_path
        read -p "포인트 좌표를 입력하세요 (예: [[100,100]]): " points
        python main.py --image "$img_path" --mode point --points "$points"
        ;;
    3)
        read -p "이미지 경로를 입력하세요: " img_path
        read -p "박스 좌표를 입력하세요 (예: [50,50,200,200]): " box
        python main.py --image "$img_path" --mode box --box "$box"
        ;;
    4)
        read -p "입력 디렉토리 경로: " input_dir
        read -p "출력 디렉토리 경로: " output_dir
        python batch_process.py --input_dir "$input_dir" --output_dir "$output_dir"
        ;;
    5)
        python webcam_demo.py --model vit_b
        ;;
    *)
        echo "종료합니다."
        ;;
esac

echo ""
echo "완료!"
