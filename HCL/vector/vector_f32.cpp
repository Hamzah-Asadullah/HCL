#include <immintrin.h>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <omp.h>

#include "./vector_vanilla.cpp"

#ifndef VECTOR_F32_CPP
#define VECTOR_F32_CPP

namespace HCL
{
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

        // AVX2_prtn
        void AVX2_prtn
        (
            vector_f32& a, const vector_f32& b, const vector_f32& c,
            __m256 (*f) (const __m256&, const __m256&), float (*fs) (const float&, const float&)
        )
        {
#ifdef DEBUG
            if ((b.size() != c.size()) || (b.size() != a.size()))
                throw std::runtime_error("HCL::vector_f32 (AVX2_prtn): Both vectors need to be of same size.");
#endif
            constexpr unsigned short batch_size = 8; // 256 / 32 = 8
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m256 vb = _mm256_load_ps(&b[i]);
                __m256 vc = _mm256_load_ps(&c[i]);
                __m256 va = f(vb, vc);
                _mm256_store_ps(&a[i], va);
            }

#pragma omp parallel for
            for (std::intmax_t i = (a.size() / batch_size) * batch_size; i < std::intmax_t(a.size()); ++i)
                a[i] = fs(b[i], c[i]);
        }

        void AVX2_prtn
        (
            vector_f32& a, const vector_f32& b, const float& c,
            __m256 (*f) (const __m256&, const __m256&), float (*fs) (const float&, const float&)
        )
        {
#ifdef DEBUG
            if ((b.size() != c.size()) || (b.size() != a.size()))
                throw std::runtime_error("HCL::vector_f32 (AVX2_prtn): Both vectors need to be of same size.");
#endif
            constexpr unsigned short batch_size = 8; // 256 / 32 = 8
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m256 vb = _mm256_load_ps(&b[i]);
                __m256 vc = _mm256_broadcast_ss(&c);
                __m256 va = f(vb, vc);
                _mm256_store_ps(&a[i], va);
            }

#pragma omp parallel for
            for (std::intmax_t i = (a.size() / batch_size) * batch_size; i < std::intmax_t(a.size()); ++i)
                a[i] = fs(b[i], c);
        }

        void AVX2_prtn
        (
            vector_f32& a, const float& b, const vector_f32& c,
            __m256 (*f) (const __m256&, const __m256&), float (*fs) (const float&, const float&)
        )
        {
#ifdef DEBUG
            if ((b.size() != c.size()) || (b.size() != a.size()))
                throw std::runtime_error("HCL::vector_f32 (AVX2_prtn): Both vectors need to be of same size.");
#endif
            constexpr unsigned short batch_size = 8; // 256 / 32 = 8
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m256 vb = _mm256_broadcast_ss(&b);
                __m256 vc = _mm256_load_ps(&c[i]);
                __m256 va = f(vb, vc);
                _mm256_store_ps(&a[i], va);
            }

#pragma omp parallel for
            for (std::intmax_t i = (a.size() / batch_size) * batch_size; i < std::intmax_t(a.size()); ++i)
                a[i] = fs(b, c[i]);
        }

        // do_AVX2_create
        template <typename T>
        vector_f32 do_AVX2_create(const vector_f32& b, const T& c, __m256 (*f) (const __m256&, const __m256&), float (*fs) (const float&, const float&))
        {
#ifdef DEBUG
            if (b.size() != c.size())
                throw std::runtime_error("HCL::vector_f32 (do_AVX2_create): Both vectors need to be of same size.");
#endif
            vector_f32 a(b.size());
            if (a.size())
                AVX2_prtn(a, b, c, f, fs);
            return a;
        }
        #elif defined(__AVX__)
        static __m128 add(const __m128& a, const __m128& b) { return _mm_add_ps(a, b); }
        static __m128 sub(const __m128& a, const __m128& b) { return _mm_sub_ps(a, b); }
        static __m128 mul(const __m128& a, const __m128& b) { return _mm_mul_ps(a, b); }
        static __m128 div(const __m128& a, const __m128& b) { return _mm_div_ps(a, b); }

        // AVX_prtn
        void AVX_prtn
        (
            vector_f32& a, const vector_f32& b, const vector_f32& c,
            __m128 (*f) (const __m128&, const __m128&), float (*fs) (const float&, const float&)
        )
        {
#ifdef DEBUG
            if ((b.size() != c.size()) || (b.size() != a.size()))
                throw std::runtime_error("HCL::vector_f32 (AVX_prtn): Both vectors need to be of same size.");
#endif

            constexpr unsigned short batch_size = 4; // 128 / 32 = 4
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m128 vb = _mm_load_ps(&b[i]);
                __m128 vc = _mm_load_ps(&c[i]);
                __m128 va = f(vb, vc);
                _mm_store_ps(&a[i], va);
            }

#pragma omp parallel for
            for (std::intmax_t i = (a.size() / batch_size) * batch_size; i < std::intmax_t(a.size()); ++i)
                a[i] = fs(b[i], c[i]);
        }

        void AVX_prtn
        (
            vector_f32& a, const vector_f32& b, const float& c,
            __m128 (*f) (const __m128&, const __m128&), float (*fs) (const float&, const float&)
        )
        {
#ifdef DEBUG
            if ((b.size() != c.size()) || (b.size() != a.size()))
                throw std::runtime_error("HCL::vector_f32 (AVX_prtn): Both vectors need to be of same size.");
#endif

            constexpr unsigned short batch_size = 4; // 128 / 32 = 4
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m128 vb = _mm_load_ps(&b[i]);
                __m128 vc = _mm_set1_ps(c);
                __m128 va = f(vb, vc);
                _mm_store_ps(&a[i], va);
            }

#pragma omp parallel for
            for (std::intmax_t i = (a.size() / batch_size) * batch_size; i < std::intmax_t(a.size()); ++i)
                a[i] = fs(b[i], c);
        }

        void AVX_prtn
        (
            vector_f32& a, const float& b, const vector_f32& c,
            __m128 (*f) (const __m128&, const __m128&), float (*fs) (const float&, const float&)
        )
        {
#ifdef DEBUG
            if ((b.size() != c.size()) || (b.size() != a.size()))
                throw std::runtime_error("HCL::vector_f32 (AVX_prtn): Both vectors need to be of same size.");
#endif

            constexpr unsigned short batch_size = 4; // 128 / 32 = 4
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m128 vb = _mm_set1_ps(b);
                __m128 vc = _mm_load_ps(&c[i]);
                __m128 va = f(vb, vc);
                _mm_store_ps(&a[i], va);
            }

#pragma omp parallel for
            for (std::intmax_t i = (a.size() / batch_size) * batch_size; i < std::intmax_t(a.size()); ++i)
                a[i] = fs(b, c[i]);
        }

        // do_AVX_create
        template <typename T>
        vector_f32 do_AVX_create(const vector_f32& b, const T& c, __m128 (*f) (const __m128&, const __m128&), float (*fs) (const float&, const float&))
        {
#ifdef DEBUG
            if (b.size() != c.size())
                throw std::runtime_error("HCL::vector_f32 (do_AVX_create): Both vectors need to be of same size.");
#endif
            vector_f32 a(b.size());
            if (a.size())
                AVX_prtn(a, b, c, f, fs);
            return a;
        }
        #endif

        // do_scalar_create
        template <typename T>
        static vector_f32 do_scalar_create(const vector_f32& b, const T& c, float (*fs) (const float&, const float&))
        {
            vector_f32 a(b.size());
            if (a.data() != nullptr)
                a.do_scalar(b, c, fs);
            return a;
        }

    public:
        using vector_vanilla<float>::vector_vanilla;

        #ifdef __AVX2__
        void operator+= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, add, add); }
        void operator-= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, sub, sub); }
        void operator*= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, mul, mul); }
        void operator/= (const vector_f32& vec) { AVX2_prtn(*this, *this, vec, div, div); }
        void operator+= (const float& x) { AVX2_prtn(*this, *this, x, add, add); }
        void operator-= (const float& x) { AVX2_prtn(*this, *this, x, sub, sub); }
        void operator*= (const float& x) { AVX2_prtn(*this, *this, x, mul, mul); }
        void operator/= (const float& x) { AVX2_prtn(*this, *this, x, div, div); }
        #elif defined(__AVX__)
        void operator+= (const vector_f32& vec) { AVX_prtn(*this, *this, vec, add, add); }
        void operator-= (const vector_f32& vec) { AVX_prtn(*this, *this, vec, sub, sub); }
        void operator*= (const vector_f32& vec) { AVX_prtn(*this, *this, vec, mul, mul); }
        void operator/= (const vector_f32& vec) { AVX_prtn(*this, *this, vec, div, div); }
        void operator+= (const float& x) { AVX_prtn(*this, *this, x, add, add); }
        void operator-= (const float& x) { AVX_prtn(*this, *this, x, sub, sub); }
        void operator*= (const float& x) { AVX_prtn(*this, *this, x, mul, mul); }
        void operator/= (const float& x) { AVX_prtn(*this, *this, x, div, div); }
        #else
        void operator+= (const vector_f32& vec) { do_scalar(*this, vec, add); }
        void operator-= (const vector_f32& vec) { do_scalar(*this, vec, sub); }
        void operator*= (const vector_f32& vec) { do_scalar(*this, vec, mul); }
        void operator/= (const vector_f32& vec) { do_scalar(*this, vec, div); }
        void operator+= (const float& x) { do_scalar(*this, x, add); }
        void operator-= (const float& x) { do_scalar(*this, x, sub); }
        void operator*= (const float& x) { do_scalar(*this, x, mul); }
        void operator/= (const float& x) { do_scalar(*this, x, div); }
        #endif

        #ifdef __AVX2__
        void sum(const vector_f32& a, const vector_f32& b) { AVX2_prtn(*this, a, b, add, add); }
        void dif(const vector_f32& a, const vector_f32& b) { AVX2_prtn(*this, a, b, sub, sub); }
        void pro(const vector_f32& a, const vector_f32& b) { AVX2_prtn(*this, a, b, mul, mul); }
        void quo(const vector_f32& a, const vector_f32& b) { AVX2_prtn(*this, a, b, div, div); }
        void sum(const vector_f32& a, const float& b) { AVX2_prtn(*this, a, b, add, add); }
        void dif(const vector_f32& a, const float& b) { AVX2_prtn(*this, a, b, sub, sub); }
        void pro(const vector_f32& a, const float& b) { AVX2_prtn(*this, a, b, mul, mul); }
        void quo(const vector_f32& a, const float& b) { AVX2_prtn(*this, a, b, div, div); }
        void dif(const float& a, const vector_f32& b) { AVX2_prtn(*this, a, b, sub, sub); }
        void quo(const float& a, const vector_f32& b) { AVX2_prtn(*this, a, b, div, div); }
        #elif defined(__AVX__)
        void sum(const vector_f32& a, const vector_f32& b) { AVX_prtn(*this, a, b, add, add); }
        void dif(const vector_f32& a, const vector_f32& b) { AVX_prtn(*this, a, b, sub, sub); }
        void pro(const vector_f32& a, const vector_f32& b) { AVX_prtn(*this, a, b, mul, mul); }
        void quo(const vector_f32& a, const vector_f32& b) { AVX_prtn(*this, a, b, div, div); }
        void sum(const vector_f32& a, const float& b) { AVX_prtn(*this, a, b, add, add); }
        void dif(const vector_f32& a, const float& b) { AVX_prtn(*this, a, b, sub, sub); }
        void pro(const vector_f32& a, const float& b) { AVX_prtn(*this, a, b, mul, mul); }
        void quo(const vector_f32& a, const float& b) { AVX_prtn(*this, a, b, div, div); }
        void dif(const float& a, const vector_f32& b) { AVX_prtn(*this, a, b, sub, sub); }
        void quo(const float& a, const vector_f32& b) { AVX_prtn(*this, a, b, div, div); }
        #else
        void sum(const vector_f32& a, const vector_f32& b) { do_scalar(a, b, add); }
        void dif(const vector_f32& a, const vector_f32& b) { do_scalar(a, b, sub); }
        void pro(const vector_f32& a, const vector_f32& b) { do_scalar(a, b, mul); }
        void quo(const vector_f32& a, const vector_f32& b) { do_scalar(a, b, div); }
        void sum(const vector_f32& a, const float& x) { do_scalar(a, x, add); }
        void dif(const vector_f32& a, const float& x) { do_scalar(a, x, sub); }
        void pro(const vector_f32& a, const float& x) { do_scalar(a, x, mul); }
        void quo(const vector_f32& a, const float& x) { do_scalar(a, x, div); }
        void dif(const float& a, const vector_f32& b) { do_scalar(a, b, sub); }
        void quo(const float& a, const vector_f32& b) { do_scalar(a, b, div); }
        #endif

        #ifdef __AVX2__
        vector_f32 operator+ (const vector_f32& vec) { return do_AVX2_create(*this, vec, add, add); }
        vector_f32 operator- (const vector_f32& vec) { return do_AVX2_create(*this, vec, sub, sub); }
        vector_f32 operator* (const vector_f32& vec) { return do_AVX2_create(*this, vec, mul, mul); }
        vector_f32 operator/ (const vector_f32& vec) { return do_AVX2_create(*this, vec, div, div); }
        vector_f32 operator+ (const float& x) { return do_AVX2_create(*this, x, add, add); }
        vector_f32 operator- (const float& x) { return do_AVX2_create(*this, x, sub, sub); }
        vector_f32 operator* (const float& x) { return do_AVX2_create(*this, x, mul, mul); }
        vector_f32 operator/ (const float& x) { return do_AVX2_create(*this, x, div, div); }
        #elif defined(__AVX__)
        vector_f32 operator+ (const vector_f32& vec) { return do_AVX_create(*this, vec, add, add); }
        vector_f32 operator- (const vector_f32& vec) { return do_AVX_create(*this, vec, sub, sub); }
        vector_f32 operator* (const vector_f32& vec) { return do_AVX_create(*this, vec, mul, mul); }
        vector_f32 operator/ (const vector_f32& vec) { return do_AVX_create(*this, vec, div, div); }
        vector_f32 operator+ (const float& x) { return do_AVX_create(*this, x, add, add); }
        vector_f32 operator- (const float& x) { return do_AVX_create(*this, x, sub, sub); }
        vector_f32 operator* (const float& x) { return do_AVX_create(*this, x, mul, mul); }
        vector_f32 operator/ (const float& x) { return do_AVX_create(*this, x, div, div); }
        #else
        vector_f32 operator+ (const vector_f32& vec) { return do_scalar_create(*this, vec, add); }
        vector_f32 operator- (const vector_f32& vec) { return do_scalar_create(*this, vec, sub); }
        vector_f32 operator* (const vector_f32& vec) { return do_scalar_create(*this, vec, mul); }
        vector_f32 operator/ (const vector_f32& vec) { return do_scalar_create(*this, vec, div); }
        vector_f32 operator+ (const float& x) { return do_scalar_create(*this, x, add); }
        vector_f32 operator- (const float& x) { return do_scalar_create(*this, x, sub); }
        vector_f32 operator* (const float& x) { return do_scalar_create(*this, x, mul); }
        vector_f32 operator/ (const float& x) { return do_scalar_create(*this, x, div); }
        #endif
    };
};

#endif