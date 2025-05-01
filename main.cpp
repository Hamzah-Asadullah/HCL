#include <iostream>
#include <chrono>

#include "./HCL/vector.cpp"

#ifdef DEBUG
#undef DEBUG
#endif

using namespace std::chrono;

/*
  !!! NOTE !!!
  This program will use ~12GB of memory to benchmark.
  Please ensure your machine has enough memory to run this.
  If you have less memory, consider lowering std::size_t elems.
*/

int main()
{
    std::size_t elems = 1024 * 1024 * 768;
    HCL::vector_f32 a(elems); a.setX(17);
    HCL::vector_f32 b(elems); b.setX(16);
    HCL::vector_f32 c(elems), d(elems);

    time_point start = high_resolution_clock::now(), end = start;
    
    // operation, one-lined but ~50% slower (due temp. copies): d = a * b + a / b;
    d  = a;
    d *= b;

    end = high_resolution_clock::now();
    std::cout
        << duration_cast<milliseconds>(end - start).count()
        << "ms.: Finished multiplication.\n";

    c  = a;
    c /= b;

    end = high_resolution_clock::now();
    std::cout
        << duration_cast<milliseconds>(end - start).count()
        << "ms.: Finished division.\n";

    d += c;

    end = high_resolution_clock::now();

    std::cout
        << duration_cast<seconds>(end- start).count()
        << "s.: Finished final on four vectors, each "
        << ((elems * 4) / 1024.f) / 1024.f
        << "MB of 32-bit floats." << std::endl;

    return 0;
}
