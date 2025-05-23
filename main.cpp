#include <iostream>
#include <chrono>
#include <string>

#include "./HCL/vector/vector_i8.cpp"
#define SCALAR_TYPE int8_t
#define VECTOR_TYPE HCL::vector_i8

#ifdef _WIN32 // For UTF
#include <windows.h>
#endif

#ifndef DEBUG
#define DEBUG
#endif

using namespace std::chrono;

// Test for HCL
// Test new features implemented in latest release
// For this release: test all new vec * scalar overloads

// All native HCL types now support:

// Performance (for all operations using mathematical expressions):

// - AVX2   + OpenMP
// - AVX    + OpenMP (fallback if no AVX2)
// - Scalar + OpenMP (fallback if no AVX)

// Operations (on both scalar value of same dtype, and vector of same dtype):

// - +=, -=, *=, /=
// - +, -, *, /
// - .sum, .dif, .pro, .quo

int main()
{
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif

    #ifdef __AVX2__
    std::cout << ">> Mode: AVX256.\n";
    #elif defined(__AVX__)
    std::cout << ">> Mode: AVX128.\n";
    #else
    std::cout << ">> Mode: Scalar.\n";
    #endif

    std::size_t elems = 0; // If that's 0 it'll prompt
    std::size_t bytes = sizeof(SCALAR_TYPE);

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

    VECTOR_TYPE vec(elems);
    if (vec.size())
        vec.setX(5.f);
    else
    {
        std::cout << ">> Failed to allocate memory, returning...";
        return 1;
    }

    // On GCC / G++ / MinGW, it'll most likely always return a thread count of 1. Can't do anything about it.
    std::cout << ">> Starting calculations using " << omp_get_num_threads() << " threads managed by OMP.\n";
    time_point start = high_resolution_clock::now(), end = start;
    vec += SCALAR_TYPE(15);
    vec -= SCALAR_TYPE(5);
    vec *= SCALAR_TYPE(3);
    vec /= SCALAR_TYPE(2);

    std::cout << "Hang on...\r";
    
    vec.sum(vec, SCALAR_TYPE(15));
    vec.dif(vec, SCALAR_TYPE(5));
    vec.pro(vec, SCALAR_TYPE(3));
    vec.quo(vec, SCALAR_TYPE(2));
    vec.dif(SCALAR_TYPE(10), vec);
    vec.quo(SCALAR_TYPE(10), vec);

    std::cout << "Almost done...\r";

    vec = vec + SCALAR_TYPE(15);
    vec = vec - SCALAR_TYPE(5);
    vec = vec * SCALAR_TYPE(3);
    vec = vec / SCALAR_TYPE(2);
    end = high_resolution_clock::now();

    std::cout
        << ">> Took "
        << duration_cast<milliseconds>(end - start).count()
        << "ms \U0001F618 (n0 = " << int(vec[0]) << ")\n";

    return 0;
}