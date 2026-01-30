import cudaq

qubit_count = 2

# Define our kerel
@cudaq.kernel
def kernel(qubit_count: int):

    qvector = cudaq.qvector(qubit_count) # Allocate

    h(qvector[0]) # Apply H gate to first qubit

    for qubit in range(qubit_count - 1):
        x.ctrl(qvector[qubit], qvector[qubit+1])

    # measure
    mz(qvector)


####################### Sample #########################
result = cudaq.sample(kernel, qubit_count)
print(result)
    
# cudaq.sample(): 커널과 그 인수를 받아서 cudaq.SampleResult 객체(사전)을 반환. 이  사전을 측정 상태의 이진 분포를 나타냄
# By dafault, sample은 1000 shots을 기준으로 돌아가며, shots_count 매개변수를 사용하여 임의 지정 가능
# ex) result = cudaq.sample(kernel, qubit_count, shots_count = 100)


# cudaq.SampleResult 객체로부터 다양한 정보를 뽑을 수 있음
print(result.most_probable())
print(result.probability(result.most_probable()))
#########################################################
