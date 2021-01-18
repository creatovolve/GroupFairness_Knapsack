This is the python implementation of the Algorithm 3 for the bound on value case in the paper "Group fairness in knapsack problems". The full version of the paper can be found [here](https://arxiv.org/pdf/2006.07832.pdf).

# Command line arguments

The code require the following command line arguments as input to the algorithm

Argument 1 (int) : number of items in each categories <br />
Argument 2 (int) : number of categories (groups) <br />
Argument 3 (float) : epsilon :- approximation and violation of fairness parameter <br />
Argument 4 (float) : Upper bound of value in each category for fairness <br />
Argument 5 (float) : Lower bound of value in each category for fairness <br />
Argument 6 (int) : Total capacity of knapsack <br />

The value and weight of each items are generated randomly between 1 to 100 by the code.

# System requirements

The code requires cuda enabled GPU. The code requires CUDA toolkit to be installed in the system. The detailed installation guide could be found [here](https://docs.nvidia.com/cuda/index.html#installation-guides).<br />

The code also requires numpy,math and numba libraries to be installed along with the python interpreter. The instructions to install numba could be found [here](https://numba.pydata.org/numba-doc/latest/user/installing.html). 

# Note

The code could be found in bound_value.py file. It was written for GeForce GTX 1080 Ti GPU. The code allocates GPU memory and forks parallel threads according to the limitations of GeForce GTX 1080 Ti GPU. If the host GPU is different, the memory allocation size and number of threads in the code might need to be modified according to the host GPU.
