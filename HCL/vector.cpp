#include <immintrin.h>
#include <thread>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>

#include "./aligned_malloc.cpp"

#ifndef VECTOR_CPP
#define VECTOR_CPP

namespace HCL
{
    template <typename T>
    class vector_vanilla
    {
    private:
        T* mem = nullptr;
        std::size_t n_elems;

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
            else
            {
                _free();
                if (elems == 0)
                    return n_elems;
            }

            mem = simple_aligned_malloc<T>(sizeof(T) * 8, sizeof(T) * elems);
            if (mem == nullptr) _free();
            else
            {
                n_elems = elems;
                setX(T(0));
            }
            return n_elems;
        }

        void setX(T x)
        {
            for (std::size_t i = 0; i < n_elems; ++i)
                mem[i] = x;
        }

        void applyFn(T (*fn) (const T&))
        {
            for (std::size_t i = 0; i < n_elems; ++i)
                mem[i] = fn(mem[i]);
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

    class vector_f64 : public vector_vanilla<double>
    {
    private:
        static __m256d add(const __m256d& a, const __m256d& b) { return _mm256_add_pd(a, b); }
        static __m256d sub(const __m256d& a, const __m256d& b) { return _mm256_sub_pd(a, b); }
        static __m256d mul(const __m256d& a, const __m256d& b) { return _mm256_mul_pd(a, b); }
        static __m256d div(const __m256d& a, const __m256d& b) { return _mm256_div_pd(a, b); }
    
        void AVX2_prtn
        (
            vector_f64& a, const vector_f64& b, const vector_f64& c,
            __m256d (*f) (const __m256d&, const __m256d&)
        )
        {
#ifdef DEBUG
            if ((b.n_elems != c.n_elems) || (b.n_elems != a.n_elems))
                throw std::runtime_error("HCL::vector_f64 (AVX2_prtn): Both vectors need to be of same length.");
#endif
            constexpr unsigned short batch_size = 4; // 256 / 64 = 4
            std::intmax_t i = 0, simd_range = a.size() - batch_size;
            // signed since under 8 elems get's negative => seg error
            // i is signed to avoid conversion on stuff like -O0

            for (; i <= simd_range; i += batch_size)
            {
                __m256d vb = _mm256_load_pd(&b[i]);
                __m256d vc = _mm256_load_pd(&c[i]);
                __m256d va = f(vb, vc);
                _mm256_store_pd(&a[i], va);
            }

            for (; i < a.size(); ++i)
                a[i] = b[i] + c[i];
        }

    public:
        using vector_vanilla<double>::vector_vanilla;    

        void operator+= (const vector_f64& vec) { AVX2_prtn(*this, *this, vec, add); }
        void operator-= (const vector_f64& vec) { AVX2_prtn(*this, *this, vec, sub); }
        void operator*= (const vector_f64& vec) { AVX2_prtn(*this, *this, vec, mul); }
        void operator/= (const vector_f64& vec) { AVX2_prtn(*this, *this, vec, div); }

        vector_f64 operator+ (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, add);
            return tmp;
        }
        vector_f64 operator- (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, sub);
            return tmp;
        }
        vector_f64 operator* (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, mul);
            return tmp;
        }
        vector_f64 operator/ (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, div);
            return tmp;
        }
    };

    class vector_f32 : public vector_vanilla<float>
    {
    private:
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
                throw std::runtime_error("HCL::vector_f64 (AVX2_prtn): Both vectors need to be of same length.");
#endif
            constexpr unsigned short batch_size = 8; // 256 / 32 = 4
            std::intmax_t i = 0, simd_range = a.size() - batch_size;

            for (; i <= simd_range; i += batch_size)
            {
                __m256 vb = _mm256_load_ps(&b[i]);
                __m256 vc = _mm256_load_ps(&c[i]);
                __m256 va = f(vb, vc);
                _mm256_store_ps(&a[i], va);
            }

            for (; i < a.size(); ++i)
                a[i] = b[i] + c[i];
        }

    public:
        using vector_vanilla<float>::vector_vanilla;    

        void operator+= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, add); }
        void operator-= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, sub); }
        void operator*= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, mul); }
        void operator/= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, div); }

        vector_f32 operator+ (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, add);
            return tmp;
        }
        vector_f32 operator- (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, sub);
            return tmp;
        }
        vector_f32 operator* (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, mul);
            return tmp;
        }
        vector_f32 operator/ (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, div);
            return tmp;
        }
    };
};

template <typename T>
std::ostream& operator<< (std::ostream& os, const HCL::vector_vanilla<T>& vec)
{
    std::size_t till = vec.size() - 1;
    if (vec.is_column)
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