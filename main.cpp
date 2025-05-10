#include <iostream>
#include <chrono>
#include <string>

#include "./HCL/vector.cpp"

#ifdef _WIN32 // For UTF
#include <windows.h>
#endif

#define TYPE float

using namespace std::chrono;

int main()
{
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif

    std::size_t elems = 0; // If that's 0 it'll prompt
    std::size_t vectors = 4;
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
                std::size_t(((std::stoull(tmp) * 1024 * 1024) / bytes) / vectors)
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

    HCL::vector_f32 vec[vectors];
    for (std::size_t v = 0; v < vectors; ++v)
    {
        if (vec[v].resize(elems) == 0) std::cout << "Failed to allocate memory for vector" << v + 1 << ", expect a crash.\n";
        vec[v].setX(5);
    }

    // On GCC / G++ / MinGW, it'll most likely always return a thread count of 1. Can't do anything about it.
    std::cout << ">> Starting calculations using " << omp_get_num_threads() << " threads managed by OMP.";
    time_point start = high_resolution_clock::now(), end = start;
    for (std::size_t v = 1; v < vectors; ++v) // v = 1 to avoid div through 0 crashes
    {
        vec[0] += vec[v];
        vec[0] *= vec[v];
        vec[0] -= vec[v];
        vec[0] /= vec[v];
    }
    end = high_resolution_clock::now();

    std::cout
        << "\r>> "
        << duration_cast<seconds>(end - start).count()
        << "s.: Finished final on " << vectors << " vectors, each "
        << ((elems * bytes) / 1024.f) / 1024.f
        << "MB of 32-bit floats \U0001F618\n";

    return 0;
}