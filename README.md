**HCL - A simple library which supports your optimizations, types, and other.**

HCL, short for `Hamzah's Computing Library` is a simple headers-only library which allows for different kind of optimizations, including:

- The use of SIMD, including AVX, AVX2, AVX512, SSE, and other
- The use of multiple threads
- The use of GPUs by manually overloading using drivers like Vulkan, CUDA, and other

This is done by a very minimal and simple concept:  
HCL has a "base" class defined in `./HCL/vector.cpp` named `HCL::vector_vanilla<T>`. Standard classes natively included in HCL, like `vector_f32` (float array) and `vector_f64` (double array) derive from this base class, which allows to add support for AVX2 and other optimizations for each type.  
This not only allows me, the developer of this library, but also you, to add support for your own type, and its optimizations, directly to the library, simply by deriving from the base class and overloading operators like `*=` efficiently.  
You can find a demo of the library using `HCL::vector_f32` (including time-based benchmarking) in the `main.cpp`.
