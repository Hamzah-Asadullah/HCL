#include <immintrin.h>
#include <thread>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <omp.h>

#include "./aligned_malloc.cpp"

#ifndef VECTOR_CPP
#define VECTOR_CPP

namespace HCL
{
    class vector_f32
    {
    private:
        float* mem = nullptr;
        std::size_t n_elems;
        bool columnvector = true;

        void add(vector_f32& a, const vector_f32& b, const vector_f32& c)
        {
#ifdef DEBUG
            if ((b.n_elems != c.n_elems) || (b.n_elems != a.n_elems))
                throw std::runtime_error("HCL::vector_f32 (add): Both vectors need to be of same length.");
#endif
            constexpr unsigned short batch_size = 8; // 256 / 32 = 8
            std::size_t i = 0, simd_range = a.n_elems - batch_size;

            for (; i <= simd_range; i += batch_size)
            {
                __m256 vb = _mm256_load_ps(&b[i]);
                __m256 vc = _mm256_load_ps(&c[i]);
                __m256 va = _mm256_add_ps(vb, vc);
                _mm256_store_ps(&a[i], va);
            }

            for (; i < a.n_elems; ++i)
                a[i] = b[i] + c[i];
        }

        void subtract(vector_f32& a, const vector_f32& b, const vector_f32& c)
        {
#ifdef DEBUG
            if ((b.n_elems != c.n_elems) || (b.n_elems != a.n_elems))
                throw std::runtime_error("HCL::vector_f32 (subtract): Both vectors need to be of same length.");
#endif
            constexpr unsigned short batch_size = 8; // 256 / 32 = 8
std::size_t i = 0, simd_range = a.n_elems - batch_size;

            for (; i <= simd_range; i += batch_size)
            {
                __m256 vb = _mm256_load_ps(&b[i]);
                __m256 vc = _mm256_load_ps(&c[i]);
                __m256 va = _mm256_sub_ps(vb, vc);
                _mm256_store_ps(&a[i], va);
            }

            for (; i < a.n_elems; ++i)
                a[i] = b[i] - c[i];
        }

        void multiply(vector_f32& a, const vector_f32& b, const vector_f32& c)
        {
#ifdef DEBUG
            if ((b.n_elems != c.n_elems) || (b.n_elems != a.n_elems))
                throw std::runtime_error("HCL::vector_f32 (multiply): Both vectors need to be of same length.");
#endif
            constexpr unsigned short batch_size = 8; // 256 / 32 = 8
std::size_t i = 0, simd_range = a.n_elems - batch_size;

            for (; i <= simd_range; i += batch_size)
            {
                __m256 vb = _mm256_load_ps(&b[i]);
                __m256 vc = _mm256_load_ps(&c[i]);
                __m256 va = _mm256_mul_ps(vb, vc);
                _mm256_store_ps(&a[i], va);
            }

            for (; i < a.n_elems; ++i)
                a[i] = b[i] * c[i];
        }

        void divide(vector_f32& a, const vector_f32& b, const vector_f32& c)
        {
#ifdef DEBUG
            if ((b.n_elems != c.n_elems) || (b.n_elems != a.n_elems))
                throw std::runtime_error("HCL::vector_f32 (divide): Both vectors need to be of same length.");
#endif
            constexpr unsigned short batch_size = 8; // 256 / 32 = 8
std::size_t i = 0, simd_range = a.n_elems - batch_size;

            for (; i <= simd_range; i += batch_size)
            {
                __m256 vb = _mm256_load_ps(&b[i]);
                __m256 vc = _mm256_load_ps(&c[i]);
                __m256 va = _mm256_div_ps(vb, vc);
                _mm256_store_ps(&a[i], va);
            }

            for (; i < a.n_elems; ++i)
                a[i] = b[i] / c[i];
        }

    public:
        vector_f32(): mem(nullptr), n_elems(0), columnvector(true) {}
        vector_f32(std::size_t elems) { resize(elems); }

        std::size_t size() const { return n_elems; }
        void transpose() { columnvector = !columnvector; }
        bool is_column() const { return columnvector; }
        const float* data() const { return &mem[0]; }

        void resize(std::size_t elems)
        {
            if (elems == n_elems) setX(0);
            else
            {
                _free();
                if (elems == 0) return;
            }

            mem = simple_aligned_malloc<float>(32, sizeof(float) * elems);
            if (mem == nullptr)
            {
                _free();
                std::cerr << ">> Failed to allocate " << (sizeof(float) * elems) / 1024 << "kB\n";
            }
            else
            {
                n_elems = elems;
                setX(float(0));
            }
        }

        void setX(float x)
        {
            for (std::size_t i = 0; i < n_elems; ++i)
                mem[i] = x;
        }

        void applyFun(float (*fun) (const float&))
        {
            for (std::size_t i = 0; i < n_elems; ++i)
                mem[i] = fun(mem[i]);
        }

        void _free()
        {
            if (mem != nullptr)
            {
                simple_aligned_free<float>(mem);
                mem = nullptr;
            }
            n_elems = 0;
        }

        float& operator[] (std::size_t i) { return mem[i]; }
        const float& operator[] (std::size_t i) const { return mem[i]; }

        void operator= (const vector_f32& vec)
        {
            if (n_elems != vec.n_elems)
                resize(vec.n_elems);
            for (std::size_t i = 0; i < n_elems; ++i)
                mem[i] = vec[i];
            columnvector = vec.columnvector;
        }

        void operator+= (const vector_f32& vec) { add(*this, *this, vec); }
        void operator-= (const vector_f32& vec) { subtract(*this, *this, vec); }
        void operator*= (const vector_f32& vec) { multiply(*this, *this, vec); }
        void operator/= (const vector_f32& vec) { divide(*this, *this, vec); }

        vector_f32 operator+ (const vector_f32& vec)
        {
            vector_f32 tmp(n_elems);
            if (tmp.mem != nullptr) add(tmp, *this, vec);
            return tmp;
        }
        vector_f32 operator- (const vector_f32& vec)
        {
            vector_f32 tmp(n_elems);
            if (tmp.mem != nullptr) subtract(tmp, *this, vec);
            return tmp;
        }
        vector_f32 operator* (const vector_f32& vec)
        {
            vector_f32 tmp(n_elems);
            if (tmp.mem != nullptr) multiply(tmp, *this, vec);
            return tmp;
        }
        vector_f32 operator/ (const vector_f32& vec)
        {
            vector_f32 tmp(n_elems);
            if (tmp.mem != nullptr) divide(tmp, *this, vec);
            return tmp;
        }

        ~vector_f32() { _free(); }
    };
};

template <typename T>
std::ostream& operator<< (std::ostream& os, const HCL::vector_f32& vec)
{
    std::size_t till = vec.size() - 1;
    if (vec.is_column())
    {
        for (std::size_t i = 0; i < till; ++i)
            os << "[ " << vec[i] << " ]\n";
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