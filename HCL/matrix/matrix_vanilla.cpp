#include <algorithm>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <omp.h>

#include "../aligned_malloc.cpp"

#ifndef MATRIX_VANILLA_CPP
#define MATRIX_VANILLA_CPP

namespace HCL
{
    struct matrix_shape
    {
        std::size_t rows = 0, cols = 0, sum = 0;

        bool is_duplicate(const matrix_shape& s) const
        {
            if ((s.rows == rows) and (s.cols == cols))
                return true;
            return false;
        }

        matrix_shape(std::size_t r, std::size_t c) { rows = r, cols = c; sum = r * c; }
        matrix_shape(): rows(0), cols(0), sum(0) {}
    };
    
    template <typename T>
    class matrix_vanilla
    {
    private:
        T* mem = nullptr;
        matrix_shape shape = { 0, 0 };

    public:
        matrix_vanilla(): mem(nullptr), shape(0, 0) {}
        matrix_vanilla(std::size_t rows, std::size_t cols) { resize(rows, cols); }
        
        // resize
        matrix_shape resize(std::size_t rows, std::size_t cols)
        {
            if ((shape.rows == rows) and (shape.cols == cols))
            {
                setX(T(0));
                return shape;
            }
            else _free();

            std::size_t block_size = std::max(sizeof(T) * 8, alignof(std::max_align_t));
                        block_size = std::max(block_size, std::size_t(32));
            mem = simple_aligned_malloc<T>(block_size, sizeof(T) * rows * cols);

            if (mem == nullptr) _free();
            else
            {
                shape.rows = rows;
                shape.cols = cols;
                shape.sum = rows * cols;
                setX(T(0));
            }

            return shape;
        }

        // transpose
        void transpose()
        {

        }

        // size data setX applyFn
        const matrix_shape& size() const { return shape; }
        const T* data() const { return mem; }
        void setX(T x) { std::fill(mem, mem + shape.sum, x); }
        void applyFn(T (*f) (const T&))
        {
            std::intmax_t elems(shape.sum);
#pragma omp parallel for
            for (std::intmax_t i = 0; i < elems; ++i)
                mem[i] = f(mem[i]);
        }

        // do_scalar
        void do_scalar(const matrix_vanilla<T>& b, const matrix_vanilla<T>& c, T (*fs) (const T&, const T&))
        {
#ifdef DEBUG
            if (not (shape.is_duplicate(b.size()) or shape.is_duplicate(c.size())))
                throw std::runtime_error("HCL::matrix_vanilla (do_scalar): Matrices a, b and c have to be of same size.");
#endif
            std::intmax_t elems = shape.sum;
#pragma omp parallel for
            for (std::intmax_t i = 0; i < elems; ++i)
                mem[i] = fs(b.mem[i], c.mem[i]);
        }

        void do_scalar(const matrix_vanilla<T>& b, const T& c, T (*fs) (const T&, const T&))
        {
#ifdef DEBUG
            if (not (shape.is_duplicate(b.size()) or shape.is_duplicate(c.size())))
                throw std::runtime_error("HCL::matrix_vanilla (do_scalar): Matrices a, b and c have to be of same size.");
#endif
            std::intmax_t elems = shape.sum;
#pragma omp parallel for
            for (std::intmax_t i = 0; i < elems; ++i)
                mem[i] = fs(b.mem[i], c);
        }

        void do_scalar(const T& b, const matrix_vanilla<T>& c, T (*fs) (const T&, const T&))
        {
#ifdef DEBUG
            if (not (shape.is_duplicate(b.size()) or shape.is_duplicate(c.size())))
                throw std::runtime_error("HCL::matrix_vanilla (do_scalar): Matrices a, b and c have to be of same size.");
#endif
            std::intmax_t elems = shape.sum;
#pragma omp parallel for
            for (std::intmax_t i = 0; i < elems; ++i)
                mem[i] = fs(b, c.mem[i]);
        }

        // _free
        void _free()
        {
            if (mem != nullptr)
            {
                simple_aligned_free<T>(mem);
                mem = nullptr;
            }
            shape.rows = 0;
            shape.cols = 0;
        }

        ~matrix_vanilla() { _free(); }

        T& operator[] (std::size_t i)
        {
#ifdef DEBUG
            if (i >= shape.sum)
                throw std::runtime_error("HCL::matrix_vanilla operator[]: (flattened) Matrix subscript out of range.")
#endif
            return mem[i];
        }

        const T& operator[] (std::size_t i) const
        {
#ifdef DEBUG
            if (i >= shape.sum)
                throw std::runtime_error("HCL::matrix_vanilla operator[]: (flattened) Matrix subscript out of range.")
#endif
            return mem[i];
        }

        T& operator() (std::size_t r, std::size_t c)
        {
#ifdef DEBUG
            if ((r >= shape.rows) or (c >= shape.cols))
                throw std::runtime_error("HCL::matrix_vanilla operator(std::size_t, std::size_t): Matrix subscripts out of range.");
#endif
            return mem[r * shape.cols + c];
        }

        const T& operator() (std::size_t r, std::size_t c) const
        {
#ifdef DEBUG
            if ((r >= shape.rows) or (c >= shape.cols))
                throw std::runtime_error("HCL::matrix_vanilla operator(std::size_t, std::size_t): Matrix subscripts out of range.");
#endif
            return mem[r * shape.cols + c];
        }

        void operator= (const matrix_vanilla<T>& mtrx)
        {
            if ((shape.rows != mtrx.size().rows) and (shape.cols != mtrx.size().cols))
                resize(mtrx.size().rows, mtrx.size().cols);
            std::copy(mtrx.data(), mtrx.data() + mtrx.size().rows * mtrx.size().cols, mem);
        }
    };
};

template <typename T>
std::ostream& operator<< (std::ostream& os, const HCL::matrix_vanilla<T>& mtrx)
{
    if ((mtrx.size().rows == 0) or (mtrx.size().cols == 0)) return os << "[ ]";
    std::size_t tillr = mtrx.size().rows - 1;
    std::size_t tillc = mtrx.size().cols - 1;

    if (mtrx.size().rows == 1)
    {
        os << "[ ";
        for (std::size_t c = 0; c < tillc; ++c)
            os << mtrx(0, c) << ", ";
        os << mtrx(0, tillc) << " ]";
        return os;
    }

    os << "[ ";
    for (std::size_t c = 0; c < tillc; ++c)
        os << mtrx(0, c) << ", ";
    os << mtrx(0, tillc) << " ]\n";

    for (std::size_t r = 1; r < tillr; ++r)
    {
        os << "| ";
        for (std::size_t c = 0; c < tillc; ++c)
            os << mtrx(r, c) << ", ";
        os << mtrx(0, tillc) << " |\n";
    }

    os << "[ ";
    for (std::size_t c = 0; c < tillc; ++c)
        os << mtrx(tillr, c) << ", ";
    os << mtrx(tillr, tillc) << " ]";

    return os;
}

#endif