import torch
import cupy
import cudaq
import cuquantum

print(f"=== 환경 검증 시작 ===")
print(f"✅ PyTorch GPU: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0)})")
print(f"✅ CuPy GPU: {cupy.cuda.runtime.getDeviceCount() > 0}")
print(f"✅ CUDA-Q Version: {cudaq.__version__}")

# cuPauliProp 0.3.0+ 임포트 경로 확인
try:
    # 최신 cuQuantum (0.3.0+) 에서는 별도 설치된 nvmath-python과 연동됩니다.
    from cuquantum import pauliprop
    print("✅ cuPauliProp: Import Successful")
except ImportError as e:
    try:
        # 특정 버전에서는 하위 경로로 접근해야 할 수도 있습니다.
        import cuquantum.pauliprop as pauliprop
        print("✅ cuPauliProp: Import Successful (via submodule)")
    except ImportError:
        print(f"❌ cuPauliProp: Import Failed ({e})")

# nvmath 확인 (cuQuantum 0.3.0+ 필수 의존성)
try:
    import nvmath
    print(f"✅ nvmath-python: Version {nvmath.__version__} detected")
except ImportError:
    print("❌ nvmath-python: Module not found")

print(f"=== 검증 완료 ===")
