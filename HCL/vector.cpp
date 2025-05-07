#include <immintrin.h>
#include <thread>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>

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

        static __m256 add(const __m256& a, const __m256& b) { return _mm256_add_ps(a, b); }
        static __m256 sub(const __m256& a, const __m256& b) { return _mm256_sub_ps(a, b); }
        static __m256 mul(const __m256& a, const __m256& b) { return _mm256_mul_ps(a, b); }
        static __m256 div(const __m256& a, const __m256& b) { return _mm256_div_ps(a, b); }

        void AVX2_prtn
        (
            vector_f32& a, const vector_f32& b, const vector_f32& c,
            __m256 (*f) (const __m256&, const __m256&)
        )
        {
#ifdef DEBUG
            if ((b.n_elems != c.n_elems) || (b.n_elems != a.n_elems))
                throw std::runtime_error("HCL::vector_f32 (AVX2_prtn): Both vectors need to be of same length.");
#endif
            constexpr unsigned short batch_size = 8; // 256 / 32 = 8
            std::intmax_t i = 0, simd_range = a.n_elems - batch_size;
            // signed since under 8 elems get's negative => seg error
            // i is signed to avoid conversion on stuff like -O0

            for (; i <= simd_range; i += batch_size)
            {
                __m256 vb = _mm256_load_ps(&b[i]);
                __m256 vc = _mm256_load_ps(&c[i]);
                __m256 va = f(vb, vc);
                _mm256_store_ps(&a[i], va);
            }

            for (; i < a.n_elems; ++i)
                a[i] = b[i] + c[i];
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

        float& operator[] (std::size_t i)
        {
#ifdef DEBUG
            if (i >= n_elems)
                throw std::runtime_error("HCL::vector_f32 (operator[]): Vector subscript out of range.");
#endif
            return mem[i];
        }
        const float& operator[] (std::size_t i) const
        {
#ifdef DEBUG
            if (i >= n_elems)
                throw std::runtime_error("HCL::vector_f32 (operator[]): Vector subscript out of range.");
#endif
            return mem[i];
        }

        void operator= (const vector_f32& vec)
        {
            if (n_elems != vec.n_elems)
                resize(vec.n_elems);
            for (std::size_t i = 0; i < n_elems; ++i)
                mem[i] = vec[i];
            columnvector = vec.columnvector;
        }

        void operator+= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, add); }
        void operator-= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, sub); }
        void operator*= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, mul); }
        void operator/= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, div); }

        vector_f32 operator+ (const vector_f32& vec)
        {
            vector_f32 tmp(n_elems);
            if (tmp.mem != nullptr) AVX2_prtn(tmp, *this, vec, add);
            return tmp;
        }
        vector_f32 operator- (const vector_f32& vec)
        {
            vector_f32 tmp(n_elems);
            if (tmp.mem != nullptr) AVX2_prtn(tmp, *this, vec, sub);
            return tmp;
        }
        vector_f32 operator* (const vector_f32& vec)
        {
            vector_f32 tmp(n_elems);
            if (tmp.mem != nullptr) AVX2_prtn(tmp, *this, vec, mul);
            return tmp;
        }
        vector_f32 operator/ (const vector_f32& vec)
        {
            vector_f32 tmp(n_elems);
            if (tmp.mem != nullptr) AVX2_prtn(tmp, *this, vec, div);
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