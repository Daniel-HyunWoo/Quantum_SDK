# Quantum_SDK

Quantum_SDK는 다양한 양자 컴퓨팅 프레임워크(CUDA-Q, PennyLane, Qiskit 등)를 활용한 예제, 실험, 어플리케이션, 데이터, 그리고 튜토리얼을 포함합니다.

## 폴더 구조

- **CUDA-Q/**: CUDA-Q 프레임워크 관련 예제, 어플리케이션, 데이터, 기본 코드 및 퀵스타트 자료
  - `Applications/`: 다양한 양자 알고리즘 및 어플리케이션 예제
  - `Basics/`: CUDA-Q의 기본적인 사용법 예제 및 코드
  - `Examples/`: CUDA-Q의 주요 기능 예제
  - `Quick Start/`: 빠른 시작을 위한 샘플 코드
  - `data/`: 실험에 사용되는 데이터셋
- **PennyLane/**: PennyLane 프레임워크 관련 예제 및 실험
  - `Optimization/`, `PennyLane-Qiskit/`, `Quantum Computing/`, `Quantum Hardware/`, `Quantum Machine Learning/`: 각 주제별 예제 및 데이터
- **Qiskit/**: Qiskit 프레임워크 관련 예제 및 실험

## 주요 파일 및 예제
- 각 프레임워크별로 Jupyter Notebook(`.ipynb`)과 Python 스크립트(`.py`) 예제가 제공됩니다.
- `data/` 폴더에는 MNIST 등 머신러닝 실험에 사용되는 데이터가 포함되어 있습니다.
- 어플리케이션 폴더에는 실제 양자 알고리즘(예: Bernstein-Vazirani, QAOA, Molecular Docking 등) 구현 예제가 있습니다.

## 시작하기
1. 각 프레임워크별 폴더로 이동하여 Jupyter Notebook 또는 Python 파일을 실행하세요.
2. 필요한 패키지는 각 예제의 설명 또는 requirements.txt(존재 시)에 따라 설치하세요.
3. 데이터가 필요한 경우, `data/` 폴더를 참고하세요.

## 기여 및 문의
- 새로운 예제, 실험, 데이터 추가를 환영합니다.
- 문의 및 이슈는 GitHub 저장소 또는 관리자에게 연락 바랍니다.

---

**Quantum_SDK**는 양자 컴퓨팅 실습과 연구를 위한 통합 환경을 제공합니다. 다양한 프레임워크의 예제를 통해 양자 알고리즘을 쉽게 실험하고 비교할 수 있습니다.

## 🚀 권장 환경 셋업 (A100 GPU 최적화)
이 프로젝트는 **NVIDIA A100 GPU** 및 **CUDA 12.x** 환경에서 가장 잘 작동하도록 설계되었습니다. (Python 3.11 권장)
PyTorch(GPU), cuQuantum, CUDA-Q, PennyLane, Qiskit의 충돌 없는 실행을 위해 제공된 스크립트를 사용하세요. Conda가 설치되어 있다면 자동으로 Conda를 사용하여 핵심 라이브러리(PyTorch, cuQuantum 등)를 우선 설치합니다.

```bash
# 실행 권한 부여
chmod +x setup_env.sh

# 환경 구축 및 패키지 설치 (Conda 자동 감지)
./setup_env.sh

# 환경 활성화 (Conda 사용 시)
conda activate quantum_env

# 환경 활성화 (Pip/Venv 사용 시)
source quantum_env/bin/activate
```
