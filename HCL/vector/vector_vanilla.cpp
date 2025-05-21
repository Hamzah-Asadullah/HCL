#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <omp.h>

#include "../aligned_malloc.cpp"

#ifndef VECTOR_VANILLA_CPP
#define VECTOR_VANILLA_CPP

namespace HCL
{
    template <typename T>
    class vector_vanilla
    {
    private:
        T* mem = nullptr;
        std::size_t n_elems = 0;

    public:
        bool is_column = true;

        vector_vanilla(): mem(nullptr), n_elems(0), is_column(true) {}
        vector_vanilla(std::size_t elems) { resize(elems); }

        std::size_t size() const { return n_elems; }
        void transpose() { is_column = !is_column; }
        const T* data() const { return mem; }

        std::size_t resize(std::size_t elems)
        {
            if (elems == n_elems)
            {
                setX(T(0));
                return n_elems;
            }
            else _free();

            std::size_t block_size = std::max(sizeof(T) * 8, alignof(std::max_align_t));
            #ifdef __AVX2__
            block_size = std::max(block_size, std::size_t(32));
            #elif defined(__AVX__)
            block_size = std::max(block_size, std::size_t(16));
            #endif
            mem = simple_aligned_malloc<T>(block_size, sizeof(T) * elems);
            if (mem == nullptr) _free();
            else
            {
                n_elems = elems;
                setX(T(0));
            }
            return n_elems;
        }

        void setX(T x) { std::fill(mem, mem + n_elems, x); }

        void applyFn(T (*fn) (const T&))
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < n_elems; ++i)
                mem[i] = fn(mem[i]);
        }

        void do_scalar(const vector_vanilla<T>& b, const vector_vanilla<T>& c, T (*fs) (const T&, const T&))
        {
#ifdef DEBUG
            if ((b.size() != c.size()) || (n_elems != c.size()))
                throw std::runtime_error("HCL::vector_vanilla `void do_scalar(...)`: Vectors a, b and c have to be of same size.");
#endif
#pragma omp parallel for
            for (std::size_t i = 0; i < n_elems; ++i)
                mem[i] = fs(b[i], c[i]);
        }

        void do_scalar(const vector_vanilla<T>& b, const T& c, T (*fs) (const T&, const T&))
        {
#ifdef DEBUG
            if (b.size() != c.size())
                throw std::runtime_error("HCL::vector_vanilla `void do_scalar(...)`: Vectors a and b have to be of same size.");
#endif
#pragma omp parallel for
            for (std::size_t i = 0; i < n_elems; ++i)
                mem[i] = fs(b[i], c);
        }

        void do_scalar(const T& b, const vector_vanilla<T>& c, T (*fs) (const T&, const T&))
        {
#ifdef DEBUG
            if (b.size() != c.size())
                throw std::runtime_error("HCL::vector_vanilla `void do_scalar(...)`: Vectors a and b have to be of same size.");
#endif
#pragma omp parallel for
            for (std::size_t i = 0; i < n_elems; ++i)
                mem[i] = fs(b, c[i]);
        }

        void _free()
        {
            if (mem != nullptr)
            {
                simple_aligned_free<T>(mem);
                mem = nullptr;
            }
            n_elems = 0;
        }

        T& operator[] (std::size_t i)
        {
#ifdef DEBUG
            if (i >= n_elems)
                throw std::runtime_error("HCL::vector_vanilla operator[]: Vector subscript out of range.");
#endif
            return mem[i];
        }

        const T& operator[] (std::size_t i) const
        {
#ifdef DEBUG
            if (i >= n_elems)
                throw std::runtime_error("HCL::vector_vanilla operator[]: Vector subscript out of range.");
#endif
            return mem[i];
        }

        void operator= (const vector_vanilla<T>& vec)
        {
            if (vec.size() != n_elems)
                resize(vec.size());

            std::copy(&(vec[0]), &(vec[n_elems]), &(mem[0]));
            is_column = vec.is_column;
        }

        ~vector_vanilla() { _free(); }
    };
};

template <typename T>
std::ostream& operator<< (std::ostream& os, const HCL::vector_vanilla<T>& vec)
{
    if (vec.size() == 0) return os;
    std::size_t till = vec.size() - 1;
    if (vec.is_column)
    {
        os << "[ " << vec[0] << " ]";
        for (std::size_t i = 1; i < till; ++i)
            os << "| " << vec[i] << " |\n";
        os << "[ " << vec[till] << " ]";
    }
    else
    {
        os << "[ ";
        for (std::size_t i = 0; i < till; ++i)
            os << vec[i] << ", ";
        os << vec[till] << " ]";
    }
    return os;
}

#endif