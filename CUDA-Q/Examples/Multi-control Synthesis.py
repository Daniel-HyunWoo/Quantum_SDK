# Out first option is to describe our general unitary by another pre-defined CUDA-Q kernel

import cudaq
from cudaq import spin

# A kernel that performs an X-gate on a provided qubit.
@cudaq.kernel
def x_kernel(qubit: cudaq.qubit):
    x(qubit)


# A kernel that will call `x_kernel` as a controlled operation.
@cudaq.kernel
def kernel(): 
    control_vector = cudaq.qvector(2) # 2개의 큐비트를 control register로 준비 |00> / qvector(N)는 N개의 큐비트를 담는 상자(레지스터)를 준비
    target = cudaq.qubit() # 타겟 큐빗 준비 |0> / qubit(N>=2) 는 불가
    x(control_vector) # |00> -> |11> 
    x(target) # |0> -> |1>
    x(control_vector[1]) # |11> -> |10>
    cudaq.control(x_kernel, control_vector, target) # control_vector가 |11>일 때 target에 x_kernel 적용. 여기서는 사실상 적용 X 

    # 그 결과 control_vector(|10>)과 target(|1>) 모두 변화 없음 -> |101>

# Alternatively, one may pass multiple arguments for control qubits or vectos to any controlled operation
@cudaq.kernel
def kernel_2():
    qvector = cudaq.qvector(3)
    x(qvector)  # |000> -> |111>
    x(qvector[1]) # |111> -> |101>
    x.ctrl([qvector[0], qvector[2]], qvector[1]) # qvector[0]과 qvector[2]가 모두 |1>일 때 qvector[1]에 x 적용. |101> -> |111>

    mz(qvector)



results = cudaq.sample(kernel)
results2 = cudaq.sample(kernel_2)
print(results)
print(results2)