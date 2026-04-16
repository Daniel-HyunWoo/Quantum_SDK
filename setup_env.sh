#!/bin/bash
set -e

ENV_NAME="quantum_env"
PYTHON_VERSION="3.11"

echo "=== Quantum SDK 환경 셋업 시작 (A100 & CUDA 12.x 최적화) ==="

# 1. Check for Conda
if command -v conda &> /dev/null; then
    echo ">> Conda가 감지되었습니다. Conda를 사용하여 핵심 패키지(PyTorch 등)를 설치합니다."
    
    # Create Conda Environment
    if conda info --envs | grep -q "$ENV_NAME"; then
        echo ">> 이미 '$ENV_NAME' 환경이 존재합니다. 업데이트를 진행합니다..."
        conda env update -n $ENV_NAME -f environment.yml --prune
    else
        echo ">> '$ENV_NAME' 환경을 생성합니다..."
        conda env create -f environment.yml
    fi
    
    echo "=== 셋업 완료 (Conda) ==="
    echo "다음 명령어를 통해 환경을 활성화하세요:"
    echo "conda activate $ENV_NAME"

else
    echo ">> Conda를 찾을 수 없습니다. 기본 venv와 pip를 사용하여 설치를 진행합니다."
    
    PYTHON_CMD="python3.11"
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo "Error: $PYTHON_CMD 이 설치되어 있지 않습니다. sudo apt install python3.11 python3.11-venv 로 먼저 설치해주세요."
        exit 1
    fi

    # Create venv
    if [ ! -d "$ENV_NAME" ]; then
        echo ">> 가상환경($ENV_NAME)을 생성합니다..."
        $PYTHON_CMD -m venv $ENV_NAME
    fi

    source $ENV_NAME/bin/activate
    pip install --upgrade pip setuptools wheel

    echo ">> requirements.txt 패키지 설치 중... (PyTorch GPU 가속 포함)"
    # PyTorch CUDA 12.1 인덱스 사용
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt

    echo "=== 셋업 완료 (Pip) ==="
    echo "다음 명령어를 통해 환경을 활성화하세요:"
    echo "source $ENV_NAME/bin/activate"
fi
