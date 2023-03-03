# Table of Contents
1. [Introduction](#Introduction)
2. [Host-Device Synchronization](#Host-Device%20Synchronization)
3. [CUDA events](#CUDA%20events)
4. [Warmup steps](#Warmup%20steps)
5. [Fixed clocks](#Fixed%20clocks)
6. [Cache flush](#Cache%20flush)
7. [Sleep / CUDA graphs](#Sleep%20/%20CUDA%20graphs)
8. [PyTorch Profiler](#PyTorch%20Profiler)

## Introduction

If we know anything of machine learning in 2023, it is this: bigger is better. Give your model more data, parameters, and compute and success is (somewhat) guaranteed.

However, larger models are memory-hungry and slow. To combat this, there is a range of  techniques that minimise training and inference costs. Some focus on efficient implementation of the Transformer architecture (FlashAttention[5], ZeroQuant[6]). Others involve algorithmic changes (Pruning, Sparsity [add references here]). Regardless of the approach, timing each operation in a computational graph is essential.

Doing so isn't trivial - there is a set of tricks needed to get accurate & repeatable results. In this blog, we present a comprehensive guide to each of these.

## Host-Device Synchronization

Our starting point is host-device synchronization. 

PyTorch executes GPU kernels asynchronously. Whilst a CUDA kernel runs on GPU, the CPU continues to queue up further kernels behind it. This prevents being bottlenecked by general overhead costs such as launching kernels and those associated with the Python interpreter.

It also has implications for timing GPU operations. A naïve approach may end up timing the kernel *launch* instead of kernel *execution*, like so:

![](_attachments/Screenshot%202023-03-03%20at%2011.50.04.png)


The common solution is to call `torch.cuda.synchronize()` before taking a timing measurement. This waits for all kernels in all CUDA streams to complete:

![](_attachments/Screenshot%202023-03-03%20at%2011.50.21.png)

Here's an example in PyTorch:

```python
import torch
from time import perf_counter

times = []
sync_times = []

# This won't time the CUDA kernel -  only the launch overhead
for _ in range(10):
    start_time = perf_counter()
    
    run_kernel()    # Some kernel
    
    end_time = perf_counter()
    times.append(end_time - start_time)
    
    
# This measures what we actually care about
for _ in range(10):
    start_time = perf_counter()
    
    run_kernel()
    torch.cuda.synchronize()
    
    end_time = perf_counter()
    sync_times.append(end_time - start_time)
```

## CUDA events
When combining explicit synchronization points with `perf_counter`, we don't just time kernel execution. We also include some overhead associated with kernel launch. In addition, using synchronization points may skew results by slowing the overall system.

[CUDA Events](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html#:~:text=CUDA%20events%20are%20synchronization%20markers,or%20exported%20to%20another%20process.) are a neat way to avoid unnecessary synchronization points and hide kernel launch overhead. Here's an example:

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

times = []
for _ in range(10):
    start_event.record()
    
    run_kernel()
    
    end_event.record()
    torch.cuda.synchronize()
    times.append(start_event.elapsed_time(end_event))
```

We begin by instantiating two `torch.cuda.Event()` objects. The `record()` method essentially puts a time stamp in the stream of kernel execution. We do so before and after the operations that we wish to time. At the end, we must include a `synchronize()` statement before running `start_time.elapsed_time(end_event)`. Omitting this means that the CPU would attempt to calculate the elapsed time before the GPU has finished its work, yielding a `RuntimeError`.

This image illustrates these ideas:

![](_attachments/Screenshot%202023-03-03%20at%2012.13.23.png)

### Warmup steps

A further improvement we can make to our above examples is to include warmup steps prior to doing timed runs. This is needed to discard overheads only incurred at the start of a training or inference run, for example:

* Optimization passes / codegen applied by PyTorch’s JIT fuser after the first few input tensors are encountered
* On-the-fly microbenchmarking carried out by torch.cudnn.benchmark when selecting optimal convolution kernel for a given input shape
* Lazy loading of kernels into the CUDA context with `CUDA_MODULE_LOADING=LAZY` & CUDA 11.7+

Here's a simple example:

```python
for _ in range(10):
    run_kernel() # don't record time

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
 
times = []
for _ in range(10):
    start_event.record()
    
    run_kernel()
    
    end_event.record()
    torch.cuda.synchronize()
    times.append(start_event.elapsed_time(end_event))
```

### Fixed clocks

So far, we have focused on making our profiling results accurate. But how can we make them consistent? The clock speed on GPUs can vary significantly according to limits on temperature and power consumption. As such, fixing the clock can substantially improve the reproducibility of our results. We have found this to be especially important for our work at Speechmatics.

One caveat is that selecting a clock speed with `nvidia-smi` doesn't guarantee that your GPU will run at the requested speed. The GPU always retains the ability to decrease the clock speed (throttling) - this is to prevent damage to the hardware. But by setting the clock speed to some rate sufficiently below the maximum, we can ensure that the level of throttling is less severe.

Here's an example of how to implement this in Python. We use a similar approach to that of OpenAI's Triton DSL [6] and Deepmind's AlphaTensor [7] repositories.

```python
import os
import subprocess

DEVICE = os.environ.get("CUDA_VISIBLE_DEVICES")
CLOCK_SPEED = 1350  # Must choose a clock speed that's supported on your device.

def set_clock_speed():
    """
    Set GPU clock speed to a specific value.
    This doesn't guarantee a fixed value due to throttling, but can help reduce variance.
    """
    process = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, shell=True)
    stdout, _ = process.communicate()
    process = subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {DEVICE}", shell=True)
    process = subprocess.run(f"sudo nvidia-smi -lgc {CLOCK_SPEED} -i {DEVICE}", shell=True)


def reset_clock_speed():
    """
    Reset GPU clock speed to default values.
    """
    subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {DEVICE}", shell=True)
    subprocess.run(f"sudo nvidia-smi -rgc -i {DEVICE}", shell=True)
```

### Cache flush

Another import consideration is to ensure that the GPU memory caches are cleared between timing calls. This avoids the possibility of repeated kernels executions exploiting cache hits and artificically reducing latency. One simple solution is to pass different input data for each pass, but we need to be careful that we are covering all bases, for example when profiling a `torch.nn.Linear` it may be insufficient to swap out the input data as some of the (static) weights could still persist in the cache across runs. 

If the input data are large the constant recreation could also slow down the dev loop, so a more robust solution is to explicitly flush the cache between passes. The example below is based on Triton DSL [6]. It works by writing sufficient data such that any existing cache lines are overwritten, as the L2 cache on Nvidia GPUs uses a write-back policy [7] this means the `zeros` data will initially be written to the L2 cache.

```python
# allocating 40MB to match L2 cache size on A100
x = torch.empty(int(40 * (1024 ** 2)), dtype=torch.int8, device='cuda')

def flush_cache():
    x.zero_() 
 ```

### Sleep / CUDA graphs

We previously saw that CUDA events hide the overhead of launching a kernel (the fixed time between the host launching a kernel and it being executed on the GPU). However, this is not a silver bullet as it makes the assumption that there is no time gap between the kernel in question and the surrounding CUDA events in the command queue. That is, it assumes the preceding CUDA event completes immediately before the kernel is due to be executed, and the following CUDA event starts as soon as the kernel is complete. When we are timing lightwight kernels that are fast to execute this assumption can break down. This can lead to spurious results which contain launch overhead in the CUDA events delta, and is illustrated in the diagram below.

Luckily there are solutions, the simplest is to apply backpressure to the command queue to ensure that the the kernel and it's events are enqueued together, rather than being executed before the next command has a chance to make it onto the queue. A naive approach to this would be to launch a sufficiently expensive kernel prior to the events/kernel we are interested in, to create a backlog. A cleaner solution would be to ask the GPU to wait for a fixed number of instruction cycles, either by using CUDA's `__nanosleep` or `torch.cuda._sleep()`.

```python
set_clock_speed()

for _ in range(10):
    run_kernel() # don't record time

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
 
times = []
for _ in range(10):
    flush_cache()
    
    torch.cuda._sleep(1_000_000)
    start_event.record()
    
    run_kernel()
    
    end_event.record()
    torch.cuda.synchronize()
    times.append(start_event.elapsed_time(end_event))
    
reset_clock_speed()
```

A second solution is to use CUDA graphs. This minimizes the launch overhead by joining a series of independent kernel launches into a single kernel. Note that we execute the target kernel multiple times within the graph capture to amortize the cost of the launch overhead.

```python
set_clock_speed()

for _ in range(10):
    run_kernel() # don't record time
    
# capture CUDA graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for _ in range(100):
        run_kernel()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
 
times = []
for _ in range(10):

    start_event.record()
    
    graph.replay()
    
    end_event.record()
    torch.cuda.synchronize()
    times.append(start_event.elapsed_time(end_event))
    
reset_clock_speed()
```

## PyTorch Profiler

## References

Lilian Weng blog post ([https://lilianweng.github.io/posts/2023-01-10-inference-optimization/](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/))  
Flash Attention  
ZeroQuant
Triton Language ([https://github.com/openai/triton/blob/ba0198326e280192bff9cd656a0a231b613901fa/python/triton/testing.py#L420](https://github.com/openai/triton/blob/ba0198326e280192bff9cd656a0a231b613901fa/python/triton/testing.py#L420))  
AlphaTensor ([https://github.com/deepmind/alphatensor/blob/1949163da3bef7e3eb268a3ac015fd1c2dbfc767/benchmarking/run_gpu_benchmark.py#L56](https://github.com/deepmind/alphatensor/blob/1949163da3bef7e3eb268a3ac015fd1c2dbfc767/benchmarking/run_gpu_benchmark.py#L56))
 Caches https://en.wikipedia.org/wiki/Cache_(computing)
