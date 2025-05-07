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

using namespace std::chrono;

int main()
{
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif

    std::size_t elems = 0; // If that's 0 it'll prompt

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
                std::size_t(((std::stoull(tmp) * 1024 * 1024) / sizeof(float)) / 4)
            );
            std::cout << ">> Set to " << elems << " numbers per vector.\n>> 0%";
        }
        catch (...)
        {
            std::cout << ">> Didn't enter a valid number \U0001F937";
            return -1;
        }
    }

    HCL::vector_f32 a(elems); a.setX(17);
    HCL::vector_f32 b(elems); b.setX(16);
    HCL::vector_f32 c(elems), d(elems);

    time_point start = high_resolution_clock::now(), end = start;
    
    // operation, one-lined but ~50% slower (due temp. copies): d = a * b + a / b;
    d  = a;
    d *= b;
    std::cout << "\r>> 33%";

    c  = a;
    c /= b;

    a.resize(0);
    b.resize(0);
    std::cout << "\r>> 67%";

    d += c;

    end = high_resolution_clock::now();
    std::cout
        << "\r>> "
        << duration_cast<seconds>(end- start).count()
        << "s.: Finished final on four vectors, each "
        << ((elems * 4) / 1024.f) / 1024.f
        << "MB of 32-bit floats.";

    return 0;
}