#include <iostream>
#include <chrono>
#include <string>

#include "./HCL/vector/vector_f64.cpp"

#ifdef _WIN32 // For UTF
#include <windows.h>
#endif

#define TYPE double

using namespace std::chrono;

// Test for HCL
// Test new features implemented in latest release
// For this release: test all new vec * scalar overloads

int main()
{
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif

    #ifdef __AVX2__
    std::cout << ">> AVX2 found, will most likely not crash.\n";
    #elif defined(__AVX__)
    std::cout << ">> AVX found, will most likely not crash.\n";
    #else
    std::cout << ">> Both AVX2 and AVX missing; falling back to scalar. Will be \"slow\".\n";
    #endif

    std::size_t elems = 0; // If that's 0 it'll prompt
    std::size_t bytes = sizeof(TYPE);

    if (elems == 0)
    {
        std::cout << ">_ Enter MB of RAM to use: ";
        std::string tmp("");
        std::getline(std::cin, tmp);
        
        try
        {
            elems = std::max
            (
                std::size_t(1),
                std::size_t((std::stoull(tmp) * 1024 * 1024) / bytes)
            );
            std::cout << ">> Set to " << elems << " numbers per vector.\n";
        }
        catch (...)
        {
            std::cout << ">> Didn't enter a valid number \U0001F937\n";
            return -1;
        }
    }

    bool thread_set = false;
    do
    {
        std::cout << "\r>_ Enter amounts of threads to use (will be managed by OMP): ";
        std::string tmp("");
        std::getline(std::cin, tmp);

        try { omp_set_num_threads(std::stoi(tmp)); thread_set = true; }
        catch (...) {}
    }
    while (!thread_set);

    HCL::vector_f64 vec(elems);

    // On GCC / G++ / MinGW, it'll most likely always return a thread count of 1. Can't do anything about it.
    std::cout << ">> Starting calculations using " << omp_get_num_threads() << " threads managed by OMP.\n";
    time_point start = high_resolution_clock::now(), end = start;
    vec += TYPE(15);
    vec -= TYPE(5);
    vec *= TYPE(3);
    vec /= TYPE(2);

    vec.sum(vec, TYPE(15));
    vec.dif(vec, TYPE(5));
    vec.pro(vec, TYPE(3));
    vec.quo(vec, TYPE(2));

    vec.dif(TYPE(10), vec);
    vec.quo(TYPE(10), vec);

    vec = vec + TYPE(15);
    vec = vec - TYPE(5);
    vec = vec * TYPE(3);
    vec = vec / TYPE(2);
    end = high_resolution_clock::now();

    std::cout
        << "\r>> Took "
        << duration_cast<milliseconds>(end - start).count()
        << "ms \U0001F618 (n0 = " << vec[0] << ")\n";

    return 0;
}