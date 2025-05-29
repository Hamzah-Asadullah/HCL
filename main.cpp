#include <iostream>
#include <string>
#include <chrono>
#include "./HCL/matrix.cpp"
#include "./HCL/vector.cpp"

using namespace std::chrono;

std::size_t my_stoi(const char* str)
{
    std::size_t v = 0;
    for (std::size_t x = 0; str[x] != '\0'; ++x)
    {
        v *= 10;
        v += str[x] - 48; // 48 = int('0')
    }
    return v;
}

std::string format(std::size_t x)
{
    std::string str = std::to_string(x);
    for (std::intmax_t pos = std::intmax_t(str.size()) - 3; pos > 0; pos -= 3)
        str.insert(pos, ",");
    return str;
}

high_resolution_clock::time_point start;
void start_timer() { start = high_resolution_clock::now(); }
void passed_test(std::size_t& tests_passed, const std::size_t& tests)
{
    std::cout
        << "\r>> Passed Test "
        << tests_passed
        << "/" << tests << " (" << std::to_string(float(tests_passed) / float(tests)) << "): "
        << duration_cast<milliseconds>(high_resolution_clock::now() - start).count()
        << "ms";
    ++tests_passed;
}

int main(int argc, const char* argv[]) // expected: filename count_elements {if -ompt: count_threads}
{
    for (int arg = 1; arg < argc; ++arg)
    {
        if (std::string(argv[arg]) == std::string("-ompt"))
        {
            if ((arg + 1) < argc)
            {
                omp_set_num_threads(my_stoi(argv[arg + 1]));
                std::cout << ">> Set OpenMP thread count to " << argv[arg + 1] << ".\n";
                std::cout << ">> OpenMP reports a max. of " << omp_get_max_threads() << " threads.\n";
            }
            else std::cout << ">> Warning: Flag '-ompt' (OpenMP Threads) announced but no value given.\n";
        }
    }
    
    #ifdef __AVX2__
    std::cout << ">> SIMD Mode: On (AVX2)\n";
    #elif defined(__AVX__)
    std::cout << ">> SIMD Mode: On (AVX1)\n";
    #else
    std::cout << ">> SIMD Mode: Off (Scalar)\n";
    #endif

    HCL::matrix_i8 a;
    HCL::vector_i8 b;
    std::size_t elems = 1;

    if (argc > 1)
        elems = my_stoi(argv[1]);
    a.resize(elems, elems);

    start_timer();
    std::size_t tests_passed = 0, tests = 6;

    passed_test(tests_passed, tests);
    a.sum(a, a); a.sum(a, 1.f);
    a.pro(a, a); a.pro(a, 1.f);
    a.dif(a, a); a.dif(a, 1.f); a.dif(1.f, a);
    a.quo(a, a); a.quo(a, 1.f); a.quo(1.f, a);
    passed_test(tests_passed, tests);
    a += a; a += 1.f;
    a -= a; a -= 1.f;
    a *= a; a *= 1.f;
    a /= a; a /= 1.f;
    passed_test(tests_passed, tests);
    a = a + a; a = a + 1.f;
    a = a - a; a = a - 1.f;
    a = a * a; a = a * 1.f;
    a = a / a; a = a / 1.f;
    passed_test(tests_passed, tests);
    a._free();
    b.resize(elems);

    b.sum(b, b); b.sum(b, 1.f);
    b.pro(b, b); b.pro(b, 1.f);
    b.dif(b, b); b.dif(b, 1.f); b.dif(1.f, b);
    b.quo(b, b); b.quo(b, 1.f); b.quo(1.f, b);
    passed_test(tests_passed, tests);
    b += b; b += 1.f;
    b -= b; b -= 1.f;
    b *= b; b *= 1.f;
    b /= b; b /= 1.f;
    passed_test(tests_passed, tests);
    a = a + a; a = a + 1.f;
    a = a - a; a = a - 1.f;
    a = a * a; a = a * 1.f;
    a = a / a; a = a / 1.f;
    passed_test(tests_passed, tests);
    
    std::cout << "\n>> Completed " << format(26 * elems * elems + 26 * elems) << " operations." << std::endl;

    return 0;
}