**HCL - A simple library which supports your optimizations, types, and other.**

HCL, short for `Hamzah's Computing Library` is a simple headers-only library which allows for different kind of optimizations, including:

- The use of SIMD, including AVX, AVX2, AVX512, SSE, and other
- The use of multiple threads
- The use of GPUs by manually overloading using drivers like Vulkan, CUDA, and other

This is done by a very minimal and simple concept:  

- HCL has a "base" class defined in `./HCL/vector.cpp` named `HCL::vector_vanilla<T>`.
- Standard classes natively included in HCL, like `vector_f32` (float array) and `vector_f64` (double array) derive from this base class.
- This allows for easy optimization for each type (i8, f32, f64) using both multi-threading (native's use OMP) and secure use of AVX2.

As of _10/05/2025_, following containers are native in HCL:

- `vector_vanilla<T>`: The base class from which all other types derive
- `vector_f64` and `vector_f32`: `double` and `float` containers optimized using AVX256 and OMP
- `vector_i8`: a `int8_t` / `unsigned char` container which does uses OMP and partially SIMD as full support for SIMD (mainly mult & div) is weird 

This allows me and you to add support for your own type, and its optimizations, directly to the library, simply by deriving from the base class and overloading operators like `*=` efficiently.  
You can find a demo of the library using `HCL::vector_i8` (including time-based benchmarking) in the `main.cpp`.

**View Counter**

![View Counter Visual Image](https://count.getloli.com/@Hamzah-Asadullah_HCL?name=Hamzah-Asadullah_HCL&theme=booru-lewd&padding=1&offset=0&align=center&scale=1&pixelated=0&darkmode=auto)
