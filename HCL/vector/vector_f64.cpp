#include <immintrin.h>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <omp.h>

#include "./vector_vanilla.cpp"

#ifndef VECTOR_F64_CPP
#define VECTOR_F64_CPP

namespace HCL
{
    class vector_f64 : public vector_vanilla<double>
    {
    private:
        static double add(const double& a, const double& b) { return a + b; }
        static double sub(const double& a, const double& b) { return a - b; }
        static double mul(const double& a, const double& b) { return a * b; }
        static double div(const double& a, const double& b) { return a / b; }

        static vector_f64 do_scalar_create
        (
            const vector_f64& b, const vector_f64& c,
            double (*fs) (const double&, const double&)
        )
        {
            vector_f64 a(b.size());
            if (a.data() != nullptr)
                a.do_scalar(b, c, fs);
            return a;    
        }

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
        void operator+= (const vector_f64& vec) { do_scalar(*this, vec, add); }
        void operator-= (const vector_f64& vec) { do_scalar(*this, vec, sub); }
        void operator*= (const vector_f64& vec) { do_scalar(*this, vec, mul); }
        void operator/= (const vector_f64& vec) { do_scalar(*this, vec, div); }
        #endif

        #ifdef __AVX2__
        void sum(const vector_f64& a, const vector_f64& b) { AVX2_prtn(*this, a, b, add, add); }
        void dif(const vector_f64& a, const vector_f64& b) { AVX2_prtn(*this, a, b, sub, sub); }
        void pro(const vector_f64& a, const vector_f64& b) { AVX2_prtn(*this, a, b, mul, mul); }
        void quo(const vector_f64& a, const vector_f64& b) { AVX2_prtn(*this, a, b, div, div); }
        #elif defined(__AVX__)
        void sum(const vector_f64& a, const vector_f64& b) { AVX_prtn(*this, a, b, add, add); }
        void dif(const vector_f64& a, const vector_f64& b) { AVX_prtn(*this, a, b, sub, sub); }
        void pro(const vector_f64& a, const vector_f64& b) { AVX_prtn(*this, a, b, mul, mul); }
        void quo(const vector_f64& a, const vector_f64& b) { AVX_prtn(*this, a, b, div, div); }
        #else
        void sum(const vector_f64& a, const vector_f64& b) { do_scalar(a, b, add); }
        void dif(const vector_f64& a, const vector_f64& b) { do_scalar(a, b, sub); }
        void pro(const vector_f64& a, const vector_f64& b) { do_scalar(a, b, mul); }
        void quo(const vector_f64& a, const vector_f64& b) { do_scalar(a, b, div); }
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
        vector_f64 operator+ (const vector_f64& vec) { return do_scalar_create(*this, vec, add); }
        vector_f64 operator- (const vector_f64& vec) { return do_scalar_create(*this, vec, sub); }
        vector_f64 operator* (const vector_f64& vec) { return do_scalar_create(*this, vec, mul); }
        vector_f64 operator/ (const vector_f64& vec) { return do_scalar_create(*this, vec, div); }
        #endif
    };
};

#endif