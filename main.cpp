#include <iostream>
#include "./HCL/matrix/matrix_vanilla.cpp"

int main()
{
    HCL::matrix_vanilla<int> a;

    a.resize(20, 10); std::cout << ">> Resized\n";
    a.setX(5);        std::cout << ">> Set 5\n";
    a.applyFn([](const int& x) { return x * 2; });
    a.do_scalar(a, a, [](const int& x, const int& y) { return x / y; });
    a.do_scalar(a, 1, [](const int& x, const int& y) { return x * y; });
    a.do_scalar(1, a, [](const int& x, const int& y) { return x / y; });

    std::cout << "First element: " << a(0, 0) << '\n';
    std::cout << a << '\n';
    return 0;
}