#include <iostream>
#include <chrono>
#include <string>

#include "./HCL/vector.cpp"

#ifdef DEBUG
#undef DEBUG
#endif

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

    HCL::vector_f32 vec[vectors];
    for (std::size_t v = 0; v < vectors; ++v)
    {
        vec[v].resize(elems);
        vec[v].setX(0.0125f); // some compilers optimize 0 away I guess
    }

    time_point start = high_resolution_clock::now(), end = start;
    for (std::size_t v = 0; v < vectors; ++v)
    {
        vec[0] += vec[v];
        vec[0] *= vec[v];
        vec[0] -= vec[v];
        vec[0] /= vec[v];

        if (v != 0)
            vec[v].resize(0);
    }
    end = high_resolution_clock::now();

    std::cout
        << "\r>> "
        << duration_cast<seconds>(end - start).count()
        << "s.: Finished final on " << vectors << " vectors, each "
        << ((elems * bytes) / 1024.f) / 1024.f
        << "MB of 32-bit floats \U0001F618";

    return 0;
}