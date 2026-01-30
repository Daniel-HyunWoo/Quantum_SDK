####################### Observe #########################
# cudaq.observe()는 커널과 그 인수, 그리고  cudaq.SpinOperator를 받아 ObserveResult 객체를 반환. 얘는 expectation value를 얻을 때 사용(ObserveResult.expectation())
# cudaq.spin 모듈을 통해서 operator를 파울리 행렬의 선형 합으로 정의할 수 있음.
# ex) cudaq.spin.i(), cudaq.spin.x(), 등등

import cudaq
from cudaq import spin

operator = spin.z(0)
print(operator)

@cudaq.kernel
def kernel():
    qubit = cudaq.qubit()
    
    h(qubit)

results = cudaq.observe(kernel, operator) # 얘의 default shots_count = 1로 거진 exact한 값임
print(results.expectation()) 

# 만약 기댓값을 구하는데 approximate 하게 구하고 싶으면 shots_count 를 사용
results = cudaq.observe(kernel, operator, shots_count = 100)
print(results.expectation()) 
#############################################################


