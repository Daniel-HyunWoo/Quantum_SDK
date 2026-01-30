import cudaq
from cudaq import spin

from typing import List, Tuple

# begin by defining Hamiltonian
hamiltonian = 5.907 + 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) - 6.125 * spin.z(1)



# we'd like to make ansatz
@cudaq.kernel
def kernel(angles: List[float]):
    qubits = cudaq.qvector(2)
    x(qubits[0])

    ry(angles[0], qubits[0])
    x.ctrl(qubits[1], qubits[0])
    # Note: the kernel must not contain measurement instruction

################ Part 1 ################
print(f"part 1")
# Last thing we need is to pick an optimizer from cudaq.optimizers
# We can optionally tune this optimizer through its initial parameters, iterations, opimization bounds, etc. Before passing it to 'cudaq.vqe'
optimizer = cudaq.optimizers.COBYLA()
# optimizer.max_iterations = ...

# pass all of that into 'cudaq.vqe', and it will automatically run our optimization loop
# rutrun a tuple of the minimized eigenvalue of our 'spin_operator' and the list of optimal variational parameters

energy, parameter = cudaq.vqe(
    kernel = kernel,
    spin_operator = hamiltonian,
    optimizer = optimizer,
    parameter_count = 1 # list of parameters has length of 1:  / VQE가 자동으로 만들고 최적화 할 파라미터 개수가 1개이며 List[float] 형태임
)

print(f"\nminimized <H> = {round(energy,16)}")
print(f"optimal theta = {round(parameter[0],16)}")
print('part 1 done\n\n')
############## Part 1 end ##############


################ Part 2 ################
print(f"part 2")
# Let's look at a moreadvanced variation of the previous example
# As an alternative to cudaq.vqe, we can also ust the cudaq.optimizers suite on its own to write custom variational algorithm routines. 
# Much of this can be slightly modified for use with third-party optimizers, such as scipy
optimizer = cudaq.optimizers.Adam()

# Since we'll use a gradient-based optimizer, we can leverage the CUDA-Q gradient helper class. It is purely optional and can be replaced with your own custom gradient routin
gradient = cudaq.gradients.CentralDifference()

def objective_function(parameter_vector: List[float],
                       hamiltonian=hamiltonian,
                       gradient_strategy=gradient,
                       kernel=kernel) -> Tuple[float, List[float]]:
    """
    Note: the objective function may also take extra arguments, provided they
    are passed into the function as default arguments in python.
    """

    # Call `cudaq.observe` on the spin operator and ansatz at the
    # optimizer provided parameters. This will allow us to easily
    # extract the expectation value of the entire system in the
    # z-basis.

    # We define the call to `cudaq.observe` here as a lambda to
    # allow it to be passed into the gradient strategy as a
    # function. If you were using a gradient-free optimizer,
    # you could purely define `cost = cudaq.observe().expectation()`.
    get_result = lambda parameter_vector: cudaq.observe(                 # parameter_vector는 이 함수가 받을 인자의 이름
        kernel, hamiltonian, parameter_vector).expectation()
    # `cudaq.observe` returns a `cudaq.ObserveResult` that holds the
    # counts dictionary and the `expectation`.
    cost = get_result(parameter_vector) # parameter-vector를 인자로 넘겨줌
    
    print(f"<H> = {cost}")
    # Compute the gradient vector using `cudaq.gradients.STRATEGY.compute()`.
    gradient_vector = gradient_strategy.compute(parameter_vector, get_result, cost)
    # parameter_vector: list[float], function: Callable, funcAtX: float) -> list[float]를 필요로 함
    # compute는 어디에서(parameter_vector) 무엇을(=미분 대상 함수 = get_results) 계산 최적화(cost) 할 것인지를 넘겨줘야 함.
    

    # Return the (cost, gradient_vector) tuple.
    return cost, gradient_vector


cudaq.set_random_seed(13)  # make repeatable
energy, parameter = optimizer.optimize(dimensions=1, # 얘는 parameter_vector의 차원을 의미함. 얘로 인해 objective_function의 parameter_vector 인자가 결정됨
                                       function=objective_function
)

print(f"\nminimized <H> = {round(energy,16)}")
print(f"optimal theta = {round(parameter[0],16)}")

print(f'part 2 done')
############## Part 2 end ##############