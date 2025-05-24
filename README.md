# HCL - Hamzah's Computing Library

![License](https://img.shields.io/badge/License-MIT-blue) 
![OpenMP](https://img.shields.io/badge/OpenMP-Enabled-green)
![AVX2](https://img.shields.io/badge/SIMD-AVX2%2FAVX-red)

A lightweight C++ library for high-performance vector operations with SIMD (AVX/AVX2) and OpenMP multi-threading support.

## Features

- 🚀 **SIMD Optimizations**: Auto-vectorized operations using AVX/AVX2 intrinsics
- ⚡ **Multi-threading**: Parallelized loops via OpenMP
- 📦 **Pre-built Types**: `vector_f32`, `vector_f64`, `vector_i8` for floats, doubles, and int8
- 🔧 **Extensible**: Derive from `vector_vanilla<T>` to create custom optimized types
- 🛠️ **Aligned Memory**: Cache-friendly allocation for SIMD operations

## Quick Start

### Compilation
```bash
g++ main.cpp -march=native -fopenmp -O3 -Wall -o main.exe
```

### Example Usage
```cpp
#include "./HCL/vector/vector_f64.cpp"

int main()
{
    HCL::vector_f64 a(1000), b(1000);
    a.setX(2.5);  // Fill with 2.5
    b.setX(1.5);
    
    a += b;        // AVX2-accelerated addition
    auto c = a * 3.14;  // Scalar multiplication
    std::cout << c;
    return 0;
}
```

## Documentation
[📚 Explore the Wiki](https://github.com/Hamzah-Asadullah/HCL/wiki) for:
- API reference
- Custom type creation guide
- Performance benchmarks
- GPU acceleration roadmap (CUDA, AMD kinda doesn't support anything non-shader)

## Structure
```
./
├── main.cpp
└── HCL/
    ├── aligned_malloc.cpp
    └── vector/
        ├── vector_vanilla.cpp  # Base class
        ├── vector_f32.cpp      # Float32 optimized
        ├── vector_f64.cpp      # Float64 optimized
        └── vector_i8.cpp       # Int8 operations
```

## Contributing
PRs welcome! See [CONTRIBUTING.md](https://github.com/Hamzah-Asadullah/HCL/wiki#contributing) for guidelines.

## License
MIT © 2025 Hamzah Asadullah

---

[![View Counter](https://count.getloli.com/@Hamzah-Asadullah_HCL?theme=booru-lewd)](https://github.com/Hamzah-Asadullah/HCL)
