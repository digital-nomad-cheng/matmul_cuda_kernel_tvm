# matmul_cuda_kernel_tvm
Generate optimized MatMul cuda kernel automatically using tvm auto schedule

## Motivation
TVM is a machine learning compiler framework for hetegeneous devices. It inherits the idea of decoupling computation and implementation. After defining the computation in TVM's, we can use TVM's primitive to schedule the computation manually like loop unroll, blocking and parallel. However, we can also utilize TVM's auto schedule to generate the optimized the code without requiring human expert knowledge. Most of the time the algorithms can generate efficient code on pair with expert optimized. 

In high performance parallel programming course, we have been asked to optimized gemm code using cuda. Then it comes to my mind why don't I try to generate cuda kernel code using TVM and embed it into my code directly? With this idea, I begin the journey experimenting on using TVM for generating cuda kernel.

In this repo I will try to use TVM's auto schedule to generate an efficient cuda matmul kernel, then I will write the kernel code to a seperate header file, and call this kernel from my host cpp code.


## Define matmul computation using TVM script
In TVM, there are many different levels of computation definition methods. Here we will use the TVM script to define the input and output tensor, and the computation abstraction on it.

```Python
M = 1024
N = 1024
K = 1024

# define computation using tvm script
@tvm.script.ir_module
class MyMatMulModule:
  @T.prim_func
  def main(A: T.Buffer[(M, K), "float32"],
           B: T.Buffer[(K, N), "float32"],
           C: T.Buffer[(M, N), "float32"],
           ):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for i, j, k in T.grid(M, N, K):
      with T.block("C"):
        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
        with T.init():
          C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```


We can check the TVM script computation abstraction after schedule using the following code:

```Python
sch_tuned.mod.show()
```

We can try to evaluate the GFLOPS using the following code:

```Python
from tvm.script.parser.tir import evaluate
num_flop = 2 * M * N * K
rt_mod = tvm.build(sch_tuned.mod, target="nvidia/nvidia-t4")
dev = tvm.cuda(0)
A_np = np.random.uniform(size=(M, K)).astype("float32")
B_np = np.random.uniform(size=(K, N)).astype("float32")
A_nd = tvm.nd.array(A_np, dev)
B_nd = tvm.nd.array(B_np, dev)
C_nd = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)
evaluator = rt_mod.time_evaluator("main", dev, number=10)
print("MetaSchedule: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
```

## Optimizing code by manually schedule
As I have said before, TVM decouples computation abstraction from schedule/implemenation. We can see TVM script as computation abstraction, and can try different schedule using the TVM API. In the jupyter notebook, I tried to schedule the matmul computation using shared memory and blocking which are two common techniques in high performance gemm optimization. Though it tooks a lot of time the maximum GFLOPS achieved manually is 1336 GFLOPS.

## Search efficient code using auto schedule
Using TVM auto schedule, we can only provide the API the target we want to run our program on. I use Google's colab, which has a nvidia-tesla-t4 for GPU runtime at the time of writing. The following code will search the optimal code on this GPU, utilizing information like register and shared memoery size. 
```Python
database = ms.tune_tir(
    mod=MyMatMulModule,
    target="nvidia/nvidia-t4", # define target type
    work_dir="./tune_tmp",
    max_trials_global=64,
    num_trials_per_iter=64,
    task_name="main"
)
sch_tuned = ms.tir_integration.compile_tir(database, MyMatMulModule, "nvidia/nvidia-t4")
```

The maximum GFLOPS achieved using auto schedule is 3107 GFLOPS. This is more than two times higher than the performance achieved manually. 
So I think next time when you want to write some cuda code, you might want try to use TVM and processor to do the heavy load.


## Export generated cuda kernel

Alright, you may be wondering how can we get the cuda kernel which defines the computation running on host. Here it comes:

```Python
# the following code will try to print the cuda kernel
print(rt_mod.imported_modules[0].get_source())
# here I will write it to a header file and call it in my main function
with open('matmul_tvm.h', 'w') as f:
  f.write(rt_mod.imported_modules[0].get_source())
```

## Embed cuda kernel in host code

I provide the main function to call the cuda kernel. It's exactly the same code I write for course assigment, but the kernel is generate using TVM.


The jupyter notebook provides a more detailed and inactive way for you to understand all this step by step.





