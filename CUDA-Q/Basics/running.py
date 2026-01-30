####################### Running on GPU #########################
# cudaq.set_target()을 통해 실행 타겟을 설정할 수 있음, nvidia gpu가 기본 값이고, 찾지 못하면 cpu-based(qpp-cpu)로 돌아감
import sys
import cudaq
import timeit

# Will time the execution of our sample call.
@cudaq.kernel
def kernel(qubit_count: int):

    qvector = cudaq.qvector(qubit_count) # Allocate

    h(qvector[0]) # Apply H gate to first qubit

    for qubit in range(qubit_count - 1):
        x.ctrl(qvector[qubit], qvector[qubit+1])

    # measure
    mz(qvector)


code_to_time = 'cudaq.sample(kernel, qubit_count, shots_count=100000)'
qubit_count = int(sys.argv[1]) if 1 < len(sys.argv) else 22 # sys.argv의 첫번째는 파일명, 두번째 부터는 스크립트 이름 뒤에 쓴 인자들. 즉 스크립트만 실행하면 qubit_count = 16, 스크립트 뒤에 숫자 하나만 붙여서 실행하면 그 숫자가 qubit_count가 됨


# Execute on CPU backend.
cudaq.set_target('qpp-cpu')
print('CPU time')  # Example: 27.57462 s.
print(timeit.timeit(stmt=code_to_time, globals=globals(), number=1))
# stmt: statement로 실행할 코드 
# globals: stmt로 지정된 코드는 독립 환경에서 실행되므로, 원래 스크립트에서 정의된 변수나 함수를 알지 못함. globals()를 통해 현재 스크립트의 변수 및 함수를 stmt 코드 실행 환경으로 가져감
# number: stmt 코드를 몇 번 실행할지 지정. 기본값은 1, 즉 한 번 실행

if cudaq.num_available_gpus() > 0:
    # Execute on GPU backend.
    cudaq.set_target('nvidia')
    print('GPU time')  # Example: 0.773286 s.
    print(timeit.timeit(stmt=code_to_time, globals=globals(), number=1))


