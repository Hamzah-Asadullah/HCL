#include <immintrin.h>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <omp.h>

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
        static double add(const double& a, const double& b) { return a + b; }
        static double sub(const double& a, const double& b) { return a - b; }
        static double mul(const double& a, const double& b) { return a * b; }
        static double div(const double& a, const double& b) { return a / b; }

        #ifdef __AVX2__
        static __m256d add(const __m256d& a, const __m256d& b) { return _mm256_add_pd(a, b); }
        static __m256d sub(const __m256d& a, const __m256d& b) { return _mm256_sub_pd(a, b); }
        static __m256d mul(const __m256d& a, const __m256d& b) { return _mm256_mul_pd(a, b); }
        static __m256d div(const __m256d& a, const __m256d& b) { return _mm256_div_pd(a, b); }
    
        void AVX2_prtn
        (
            vector_f64& a, const vector_f64& b, const vector_f64& c,
            __m256d (*f) (const __m256d&, const __m256d&),
            double (*fs) (const double&, const double&)
        )
        {
#ifdef DEBUG
            if ((b.n_elems != c.n_elems) || (b.n_elems != a.n_elems))
                throw std::runtime_error("HCL::vector_f64 (AVX2_prtn): Both vectors need to be of same length.");
#endif
            constexpr unsigned short batch_size = 4; // 256 / 64 = 4
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m256d vb = _mm256_load_pd(&b[i]);
                __m256d vc = _mm256_load_pd(&c[i]);
                __m256d va = f(vb, vc);
                _mm256_store_pd(&a[i], va);
            }

#pragma omp parallel for
            for (std::intmax_t i = (a.size() / batch_size) * batch_size; i < std::intmax_t(a.size()); ++i)
                a[i] = fs(b[i], c[i]);
        }
        #elif defined(__AVX__)
        static __m128d add(const __m128d& a, const __m128d& b) { return _mm_add_pd(a, b); }
        static __m128d sub(const __m128d& a, const __m128d& b) { return _mm_sub_pd(a, b); }
        static __m128d mul(const __m128d& a, const __m128d& b) { return _mm_mul_pd(a, b); }
        static __m128d div(const __m128d& a, const __m128d& b) { return _mm_div_pd(a, b); }
    
        void AVX_prtn
        (
            vector_f64& a, const vector_f64& b, const vector_f64& c,
            __m128d (*f) (const __m128d&, const __m128d&),
            double (*fs) (const double&, const double&)
        )
        {
#ifdef DEBUG
            if ((b.n_elems != c.n_elems) || (b.n_elems != a.n_elems))
                throw std::runtime_error("HCL::vector_f64 (AVX_prtn): Both vectors need to be of same length.");
#endif
            constexpr unsigned short batch_size = 2; // 128 / 64 = 4
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m128d vb = _mm_load_pd(&b[i]);
                __m128d vc = _mm_load_pd(&c[i]);
                __m128d va = f(vb, vc);
                _mm_store_pd(&a[i], va);
            }

#pragma omp parallel for
            for (std::size_t i = (a.size() / batch_size) * batch_size; i < a.size(); ++i)
                a[i] = fs(b[i], c[i]);
        }
        #endif

    public:
        using vector_vanilla<double>::vector_vanilla;    

        #ifdef __AVX2__
        void operator+= (const vector_f64& vec) { AVX2_prtn(*this, *this, vec, add, add); }
        void operator-= (const vector_f64& vec) { AVX2_prtn(*this, *this, vec, sub, sub); }
        void operator*= (const vector_f64& vec) { AVX2_prtn(*this, *this, vec, mul, mul); }
        void operator/= (const vector_f64& vec) { AVX2_prtn(*this, *this, vec, div, div); }
        #elif defined(__AVX__)
        void operator+= (const vector_f64& vec) { AVX_prtn(*this, *this, vec, add, add); }
        void operator-= (const vector_f64& vec) { AVX_prtn(*this, *this, vec, sub, sub); }
        void operator*= (const vector_f64& vec) { AVX_prtn(*this, *this, vec, mul, mul); }
        void operator/= (const vector_f64& vec) { AVX_prtn(*this, *this, vec, div, div); }
        #else
        void operator+= (const vector_f64& vec)
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < vec.size(); ++i)
                (*this)[i] += vec[i];
        }
        void operator-= (const vector_f64& vec)
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < vec.size(); ++i)
                (*this)[i] -= vec[i];
        }
        void operator*= (const vector_f64& vec)
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < vec.size(); ++i)
                (*this)[i] *= vec[i];
        }
        void operator/= (const vector_f64& vec)
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < vec.size(); ++i)
                (*this)[i] /= vec[i];
        }
        #endif

        #ifdef __AVX2__
        vector_f64 operator+ (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, add, add);
            return tmp;
        }
        vector_f64 operator- (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, sub, sub);
            return tmp;
        }
        vector_f64 operator* (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, mul, mul);
            return tmp;
        }
        vector_f64 operator/ (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, div, div);
            return tmp;
        }
        #elif defined(__AVX__)
        vector_f64 operator+ (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (tmp.data() != nullptr) AVX_prtn(tmp, *this, vec, add, add);
            return tmp;
        }
        vector_f64 operator- (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (tmp.data() != nullptr) AVX_prtn(tmp, *this, vec, sub, sub);
            return tmp;
        }
        vector_f64 operator* (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (tmp.data() != nullptr) AVX_prtn(tmp, *this, vec, mul, mul);
            return tmp;
        }
        vector_f64 operator/ (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (tmp.data() != nullptr) AVX_prtn(tmp, *this, vec, div, div);
            return tmp;
        }
        #else
        vector_f64 operator+ (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (vec.data() != nullptr)
#pragma omp parallel for
                for (std::size_t i = 0; i < tmp.size(); ++i)
                    tmp[i] = (*this)[i] + vec[i];
            return tmp;
        }
        vector_f64 operator- (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (vec.data() != nullptr)
#pragma omp parallel for
                for (std::size_t i = 0; i < tmp.size(); ++i)
                    tmp[i] = (*this)[i] - vec[i];
            return tmp;
        }
        vector_f64 operator* (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (vec.data() != nullptr)
#pragma omp parallel for
                for (std::size_t i = 0; i < tmp.size(); ++i)
                    tmp[i] = (*this)[i] * vec[i];
            return tmp;
        }
        vector_f64 operator/ (const vector_f64& vec)
        {
            vector_f64 tmp(size());
            if (vec.data() != nullptr)
#pragma omp parallel for
                for (std::size_t i = 0; i < tmp.size(); ++i)
                    tmp[i] = (*this)[i] / vec[i];
            return tmp;
        }
        #endif
    };

    class vector_f32 : public vector_vanilla<float>
    {
    private:
        static float add(const float& a, const float& b) { return a + b; }
        static float sub(const float& a, const float& b) { return a - b; }
        static float mul(const float& a, const float& b) { return a * b; }
        static float div(const float& a, const float& b) { return a / b; }

        #ifdef __AVX2__
        static __m256 add(const __m256& a, const __m256& b) { return _mm256_add_ps(a, b); }
        static __m256 sub(const __m256& a, const __m256& b) { return _mm256_sub_ps(a, b); }
        static __m256 mul(const __m256& a, const __m256& b) { return _mm256_mul_ps(a, b); }
        static __m256 div(const __m256& a, const __m256& b) { return _mm256_div_ps(a, b); }
    
        void AVX2_prtn
        (
            vector_f32& a, const vector_f32& b, const vector_f32& c,
            __m256 (*f) (const __m256&, const __m256&),
            float (*fs) (const float&, const float&)
        )
        {
#ifdef DEBUG
            if ((b.n_elems != c.n_elems) || (b.n_elems != a.n_elems))
                throw std::runtime_error("HCL::vector_f32 (AVX2_prtn): Both vectors need to be of same length.");
#endif

            constexpr unsigned short batch_size = 8; // 256 / 32 = 8
            std::size_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m256 vb = _mm256_load_ps(&b[i]);
                __m256 vc = _mm256_load_ps(&c[i]);
                __m256 va = f(vb, vc);
                _mm256_store_ps(&a[i], va);
            }

#pragma omp parallel for
            for (std::size_t i = (a.size() / batch_size) * batch_size; i < a.size(); ++i)
                a[i] = fs(b[i], c[i]);
        }
        #elif defined(__AVX__)
        static __m128 add(const __m128& a, const __m128& b) { return _mm_add_ps(a, b); }
        static __m128 sub(const __m128& a, const __m128& b) { return _mm_sub_ps(a, b); }
        static __m128 mul(const __m128& a, const __m128& b) { return _mm_mul_ps(a, b); }
        static __m128 div(const __m128& a, const __m128& b) { return _mm_div_ps(a, b); }

        void AVX_prtn
        (
            vector_f32& a, const vector_f32& b, const vector_f32& c,
            __m128 (*f) (const __m128&, const __m128&),
            float (*fs) (const float&, const float&)
        )
        {
#ifdef DEBUG
            if ((b.n_elems != c.n_elems) || (b.n_elems != a.n_elems))
                throw std::runtime_error("HCL::vector_f32 (AVX_prtn): Both vectors need to be of same length.");
#endif

            constexpr unsigned short batch_size = 4; // 128 / 32 = 4
            std::size_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m128 vb = _mm_load_ps(&b[i]);
                __m128 vc = _mm_load_ps(&c[i]);
                __m128 va = f(vb, vc);
                _mm_store_ps(&a[i], va);
            }

#pragma omp parallel for
            for (std::size_t i = (a.size() / batch_size) * batch_size; i < a.size(); ++i)
                a[i] = fs(b[i], c[i]);
        }
        #endif

    public:
        using vector_vanilla<float>::vector_vanilla;

        #ifdef __AVX2__
        void operator+= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, add, add); }
        void operator-= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, sub, sub); }
        void operator*= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, mul, mul); }
        void operator/= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, div, div); }
        #elif defined(__AVX__)
        void operator+= (const vector_f32& vec) { AVX_prtn(*this, *this, vec, add, add); }
        void operator-= (const vector_f32& vec) { AVX_prtn(*this, *this, vec, sub, sub); }
        void operator*= (const vector_f32& vec) { AVX_prtn(*this, *this, vec, mul, mul); }
        void operator/= (const vector_f32& vec) { AVX_prtn(*this, *this, vec, div, div); }
        #else
        void operator+= (const vector_f32& vec)
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < vec.size(); ++i)
                (*this)[i] += vec[i];
        }
        void operator-= (const vector_f32& vec)
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < vec.size(); ++i)
                (*this)[i] -= vec[i];
        }
        void operator*= (const vector_f32& vec)
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < vec.size(); ++i)
                (*this)[i] *= vec[i];
        }
        void operator/= (const vector_f32& vec)
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < vec.size(); ++i)
                (*this)[i] /= vec[i];
        }
        #endif

        #ifdef __AVX2__
        vector_f32 operator+ (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, add, add);
            return tmp;
        }
        vector_f32 operator- (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, sub, sub);
            return tmp;
        }
        vector_f32 operator* (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, mul, mul);
            return tmp;
        }
        vector_f32 operator/ (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, div, div);
            return tmp;
        }
        #elif defined(__AVX__)
        vector_f32 operator+ (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr) AVX_prtn(tmp, *this, vec, add, add);
            return tmp;
        }
        vector_f32 operator- (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr) AVX_prtn(tmp, *this, vec, sub, sub);
            return tmp;
        }
        vector_f32 operator* (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr) AVX_prtn(tmp, *this, vec, mul, mul);
            return tmp;
        }
        vector_f32 operator/ (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr) AVX_prtn(tmp, *this, vec, div, div);
            return tmp;
        }
        #else
        vector_f32 operator+ (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr)
#pragma omp parallel for
                for (std::size_t i = 0; i < tmp.size(); ++i)
                    tmp[i] = (*this)[i] + vec[i];
            return tmp;
        }
        vector_f32 operator- (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr)
#pragma omp parallel for
                for (std::size_t i = 0; i < tmp.size(); ++i)
                    tmp[i] = (*this)[i] - vec[i];
            return tmp;
        }
        vector_f32 operator* (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr)
#pragma omp parallel for
                for (std::size_t i = 0; i < tmp.size(); ++i)
                    tmp[i] = (*this)[i] * vec[i];
            return tmp;
        }
        vector_f32 operator/ (const vector_f32& vec)
        {
            vector_f32 tmp(size());
            if (tmp.data() != nullptr)
#pragma omp parallel for
                for (std::size_t i = 0; i < tmp.size(); ++i)
                    tmp[i] = (*this)[i] / vec[i];
            return tmp;
        }
        #endif
    };

    class vector_i8 : public vector_vanilla<int8_t>
    {
    private:
        static int8_t add(const int8_t& a, const int8_t& b) { return a + b; }
        static int8_t sub(const int8_t& a, const int8_t& b) { return a - b; }
        static int8_t mul(const int8_t& a, const int8_t& b) { return a * b; }
        static int8_t div(const int8_t& a, const int8_t& b) { return a / b; }

        #ifdef __AVX2__
        static __m256i add(const __m256i& a, const __m256i& b) { return _mm256_add_epi8(a, b); }
        static __m256i sub(const __m256i& a, const __m256i& b) { return _mm256_sub_epi8(a, b); }

        void AVX2_prtn
        (
            vector_i8& a, const vector_i8& b, const vector_i8& c,
            __m256i (*f) (const __m256i&, const __m256i&),
            int8_t (*fs) (const int8_t&, const int8_t&)
        )
        {
#ifdef DEBUG
            if ((b.n_elems != c.n_elems) || (b.n_elems != a.n_elems))
                throw std::runtime_error("HCL::vector_i8 (AVX2_prtn): Both vectors need to be of same length.");
#endif
            constexpr unsigned short batch_size = 32;
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m256i vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(&b[i]));
                __m256i vc = _mm256_load_si256(reinterpret_cast<const __m256i*>(&c[i]));
                __m256i va = f(vb, vc);
                _mm256_store_si256(reinterpret_cast<__m256i*>(&a[i]), va);
            }

#pragma omp parallel for
            for (std::intmax_t i = (a.size() / batch_size) * batch_size; i < std::intmax_t(a.size()); ++i)
                a[i] = fs(b[i], c[i]);
        }
        #elif defined(__AVX__)
        static __m128i add(const __m128i& a, const __m128i& b) { return _mm_add_epi8(a, b); }
        static __m128i sub(const __m128i& a, const __m128i& b) { return _mm_sub_epi8(a, b); }

        void AVX_prtn
        (
            vector_i8& a, const vector_i8& b, const vector_i8& c,
            __m128i (*f) (const __m128i&, const __m128i&),
            int8_t (*fs) (const int8_t&, const int8_t&)
        )
        {
#ifdef DEBUG
            if ((b.n_elems != c.n_elems) || (b.n_elems != a.n_elems))
                throw std::runtime_error("HCL::vector_i8 (AVX_prtn): Both vectors need to be of same length.");
#endif
            constexpr unsigned short batch_size = 16;
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i*>(&b[i]));
                __m128i vc = _mm_load_si128(reinterpret_cast<const __m128i*>(&c[i]));
                __m128i va = f(vb, vc);
                _mm_store_si128(reinterpret_cast<__m128i*>(&a[i]), va);
            }

#pragma omp parallel for
            for (std::intmax_t i = (a.size() / batch_size) * batch_size; i < std::intmax_t(a.size()); ++i)
                a[i] = fs(b[i], c[i]);
        }
        #endif

    public:
        using vector_vanilla<int8_t>::vector_vanilla;

        #ifdef __AVX2__
        void operator+= (const vector_i8& vec) { AVX2_prtn(*this, *this, vec, add, add); }
        void operator-= (const vector_i8& vec) { AVX2_prtn(*this, *this, vec, sub, sub); }
        #elif defined(__AVX__)
        void operator+= (const vector_i8& vec) { AVX_prtn(*this, *this, vec, add, add); }
        void operator-= (const vector_i8& vec) { AVX_prtn(*this, *this, vec, sub, sub); }
        #else
        void operator+= (const vector_i8& vec)
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < vec.size(); ++i)
                (*this)[i] *= vec[i];
        }
        void operator-= (const vector_i8& vec)
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < vec.size(); ++i)
                (*this)[i] /= vec[i];
        }
        #endif

        void operator*= (const vector_i8& vec)
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < vec.size(); ++i)
                (*this)[i] *= vec[i];
        }
        void operator/= (const vector_i8& vec)
        {
#pragma omp parallel for
            for (std::size_t i = 0; i < vec.size(); ++i)
                (*this)[i] /= vec[i];
        }

        #ifdef __AVX2__
        vector_i8 operator+ (const vector_i8& vec)
        {
            vector_i8 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, add, add);
            return tmp;
        }
        vector_i8 operator- (const vector_i8& vec)
        {
            vector_i8 tmp(size());
            if (tmp.data() != nullptr) AVX2_prtn(tmp, *this, vec, sub, sub);
            return tmp;
        }
        #elif defined(__AVX__)
        vector_i8 operator+ (const vector_i8& vec)
        {
            vector_i8 tmp(size());
            if (tmp.data() != nullptr) AVX_prtn(tmp, *this, vec, add, add);
            return tmp;
        }
        vector_i8 operator- (const vector_i8& vec)
        {
            vector_i8 tmp(size());
            if (tmp.data() != nullptr) AVX_prtn(tmp, *this, vec, sub, sub);
            return tmp;
        }
        #else
        vector_i8 operator+ (const vector_i8& vec)
        {
            vector_i8 tmp(size());
            if (tmp.data() != nullptr)
#pragma omp parallel for
                for (std::size_t i = 0; i < vec.size(); ++i)
                    tmp[i] = (*this)[i] + vec[i];
            return tmp;
        }
        vector_i8 operator- (const vector_i8& vec)
        {
            vector_i8 tmp(size());
            if (tmp.data() != nullptr)
#pragma omp parallel for
                for (std::size_t i = 0; i < vec.size(); ++i)
                    tmp[i] = (*this)[i] - vec[i];
            return tmp;
        }
        #endif

        vector_i8 operator* (const vector_i8& vec)
        {
            vector_i8 tmp(size());
            if (tmp.data() != nullptr)
#pragma omp parallel for
                for (std::size_t i = 0; i < vec.size(); ++i)
                    tmp[i] = (*this)[i] * vec[i];
            return tmp;
        }
        vector_i8 operator/ (const vector_i8& vec)
        {
            vector_i8 tmp(size());
            if (tmp.data() != nullptr)
#pragma omp parallel for
                for (std::size_t i = 0; i < vec.size(); ++i)
                    tmp[i] = (*this)[i] / vec[i];
            return tmp;
        }
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
