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
    class vector_i8 : public vector_vanilla<int8_t>
    {
    private:
        static int8_t add(const int8_t& a, const int8_t& b) { return a + b; }
        static int8_t sub(const int8_t& a, const int8_t& b) { return a - b; }
        static int8_t mul(const int8_t& a, const int8_t& b) { return a * b; }
        static int8_t div(const int8_t& a, const int8_t& b) { return a / b; }

        static vector_i8 do_scalar_create
        (
            const vector_i8& b, const vector_i8& c,
            int8_t (*fs) (const int8_t&, const int8_t&)
        )
        {
            vector_i8 a(b.size());
            if (a.data() != nullptr)
                a.do_scalar(b, c, fs);
            return a;    
        }

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
        void operator+= (const vector_i8& vec) { do_scalar(*this, vec, add); }
        void operator-= (const vector_i8& vec) { do_scalar(*this, vec, sub); }
        #endif
        void operator*= (const vector_i8& vec) { do_scalar(*this, vec, mul); }
        void operator/= (const vector_i8& vec) { do_scalar(*this, vec, div); }

        #ifdef __AVX2__
        void sum(const vector_i8& a, const vector_i8& b) { AVX2_prtn(*this, a, b, add, add); }
        void dif(const vector_i8& a, const vector_i8& b) { AVX2_prtn(*this, a, b, sub, sub); }
        #elif defined(__AVX__)
        void sum(const vector_i8& a, const vector_i8& b) { AVX_prtn(*this, a, b, add, add); }
        void dif(const vector_i8& a, const vector_i8& b) { AVX_prtn(*this, a, b, sub, sub); }
        #else
        void sum(const vector_i8& a, const vector_i8& b) { do_scalar(a, b, add); }
        void dif(const vector_i8& a, const vector_i8& b) { do_scalar(a, b, sub); }
        #endif
        void pro(const vector_i8& a, const vector_i8& b) { do_scalar(a, b, mul); }
        void quo(const vector_i8& a, const vector_i8& b) { do_scalar(a, b, div); }

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
        vector_i8 operator+ (const vector_i8& vec) { return do_scalar_create(*this, vec, add); }
        vector_i8 operator- (const vector_i8& vec) { return do_scalar_create(*this, vec, sub); }
        #endif

        vector_i8 operator* (const vector_i8& vec) { return do_scalar_create(*this, vec, mul); }
        vector_i8 operator/ (const vector_i8& vec) { return do_scalar_create(*this, vec, div); }
    };
};

#endif