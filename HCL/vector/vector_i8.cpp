#include <immintrin.h>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <omp.h>

#include "./vector_vanilla.cpp"

#ifndef VECTOR_I8_CPP
#define VECTOR_I8_CPP

namespace HCL
{
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

        // AVX2_prtn
        void AVX2_prtn
        (
            vector_i8& a, const vector_i8& b, const vector_i8& c,
            __m256i (*f) (const __m256i&, const __m256i&), int8_t (*fs) (const int8_t&, const int8_t&)
        )
        {
#ifdef DEBUG
            if (b.size() != c.size())
                throw std::runtime_error("HCL::vector_i8 (AVX2_prtn): Both vectors need to be of same size.");
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

        void AVX2_prtn
        (
            vector_i8& a, const vector_i8& b, const int8_t& c,
            __m256i (*f) (const __m256i&, const __m256i&), int8_t (*fs) (const int8_t&, const int8_t&)
        )
        {
#ifdef DEBUG
            if (b.size() != c.size())
                throw std::runtime_error("HCL::vector_i8 (AVX2_prtn): Both vectors need to be of same size.");
#endif
            constexpr unsigned short batch_size = 32;
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m256i vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(&b[i]));
                __m256i vc = _mm256_set1_epi8(c);
                __m256i va = f(vb, vc);
                _mm256_store_si256(reinterpret_cast<__m256i*>(&a[i]), va);
            }

#pragma omp parallel for
            for (std::intmax_t i = (a.size() / batch_size) * batch_size; i < std::intmax_t(a.size()); ++i)
                a[i] = fs(b[i], c);
        }

        void AVX2_prtn
        (
            vector_i8& a, const int8_t& b, const vector_i8& c,
            __m256i (*f) (const __m256i&, const __m256i&), int8_t (*fs) (const int8_t&, const int8_t&)
        )
        {
#ifdef DEBUG
            if (b.size() != c.size())
                throw std::runtime_error("HCL::vector_i8 (AVX2_prtn): Both vectors need to be of same size.");
#endif
            constexpr unsigned short batch_size = 32;
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m256i vb = _mm256_set1_epi8(b);
                __m256i vc = _mm256_load_si256(reinterpret_cast<const __m256i*>(&c[i]));
                __m256i va = f(vb, vc);
                _mm256_store_si256(reinterpret_cast<__m256i*>(&a[i]), va);
            }

#pragma omp parallel for
            for (std::intmax_t i = (a.size() / batch_size) * batch_size; i < std::intmax_t(a.size()); ++i)
                a[i] = fs(b, c[i]);
        }

        // do_AVX_create
        template <typename T>
        vector_i8 do_AVX2_create(const vector_i8& b, const T& c, __m256i (*f) (const __m256i&, const __m256i&), int8_t (*fs) (const int8_t&, const int8_t&))
        {
            vector_i8 a(b.size());
            if (a.size())
                AVX2_prtn(a, b, c, f, fs);
            return a;
        }
        #elif defined(__AVX__)
        static __m128i add(const __m128i& a, const __m128i& b) { return _mm_add_epi8(a, b); }
        static __m128i sub(const __m128i& a, const __m128i& b) { return _mm_sub_epi8(a, b); }

        // AVX_prtn
        void AVX_prtn
        (
            vector_i8& a, const vector_i8& b, const vector_i8& c,
            __m128i (*f) (const __m128i&, const __m128i&), int8_t (*fs) (const int8_t&, const int8_t&)
        )
        {
#ifdef DEBUG
            if (b.size() != c.size())
                throw std::runtime_error("HCL::vector_i8 (AVX_prtn): Both vectors need to be of same size.");
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

        void AVX_prtn
        (
            vector_i8& a, const vector_i8& b, const int8_t& c,
            __m128i (*f) (const __m128i&, const __m128i&), int8_t (*fs) (const int8_t&, const int8_t&)
        )
        {
#ifdef DEBUG
            if (b.size() != c.size())
                throw std::runtime_error("HCL::vector_i8 (AVX_prtn): Both vectors need to be of same size.");
#endif
            constexpr unsigned short batch_size = 16;
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i*>(&b[i]));
                __m128i vc = _mm_set1_epi8(c);
                __m128i va = f(vb, vc);
                _mm_store_si128(reinterpret_cast<__m128i*>(&a[i]), va);
            }

#pragma omp parallel for
            for (std::intmax_t i = (a.size() / batch_size) * batch_size; i < std::intmax_t(a.size()); ++i)
                a[i] = fs(b[i], c);
        }

        void AVX_prtn
        (
            vector_i8& a, const int8_t& b, const vector_i8& c,
            __m128i (*f) (const __m128i&, const __m128i&), int8_t (*fs) (const int8_t&, const int8_t&)
        )
        {
#ifdef DEBUG
            if (b.size() != c.size())
                throw std::runtime_error("HCL::vector_i8 (AVX_prtn): Both vectors need to be of same size.");
#endif
            constexpr unsigned short batch_size = 16;
            std::intmax_t simd_range = std::intmax_t(a.size()) - batch_size;

#pragma omp parallel for
            for (std::intmax_t i = 0; i <= simd_range; i += batch_size)
            {
                __m128i vb = _mm_set1_epi8(b);
                __m128i vc = _mm_load_si128(reinterpret_cast<const __m128i*>(&c[i]));
                __m128i va = f(vb, vc);
                _mm_store_si128(reinterpret_cast<__m128i*>(&a[i]), va);
            }

#pragma omp parallel for
            for (std::intmax_t i = (a.size() / batch_size) * batch_size; i < std::intmax_t(a.size()); ++i)
                a[i] = fs(b, c[i]);
        }

        // do_AVX_create
        template <typename T>
        vector_i8 do_AVX_create(const vector_i8& b, const T& c, __m128i (*f) (const __m128i&, const __m128i&), int8_t (*fs) (const int8_t&, const int8_t&))
        {
#ifdef DEBUG
            if (b.size() != c.size())
                throw std::runtime_error("HCL::vector_i8 (do_AVX_create): Both vectors need to be of same size.");
#endif
            vector_i8 a(b.size());
            if (a.size())
                AVX_prtn(a, b, c, f, fs);
            return a;
        }
        #endif

        // do_scalar_create
        template <typename T>
        static vector_i8 do_scalar_create(const vector_i8& b, const T& c, int8_t (*fs) (const int8_t&, const int8_t&))
        {
            vector_i8 a(b.size());
            if (a.data() != nullptr)
                a.do_scalar(b, c, fs);
            return a;    
        }

    public:
        using vector_vanilla<int8_t>::vector_vanilla;

        #ifdef __AVX2__
        void operator+= (const vector_i8& vec) { AVX2_prtn(*this, *this, vec, add, add); }
        void operator-= (const vector_i8& vec) { AVX2_prtn(*this, *this, vec, sub, sub); }
        void operator+= (const int8_t& x) { AVX2_prtn(*this, *this, x, add, add); }
        void operator-= (const int8_t& x) { AVX2_prtn(*this, *this, x, sub, sub); }
        #elif defined(__AVX__)
        void operator+= (const vector_i8& vec) { AVX_prtn(*this, *this, vec, add, add); }
        void operator-= (const vector_i8& vec) { AVX_prtn(*this, *this, vec, sub, sub); }
        void operator+= (const int8_t& vec) { AVX_prtn(*this, *this, vec, add, add); }
        void operator-= (const int8_t& vec) { AVX_prtn(*this, *this, vec, sub, sub); }
        #else
        void operator+= (const vector_i8& vec) { do_scalar(*this, vec, add); }
        void operator-= (const vector_i8& vec) { do_scalar(*this, vec, sub); }
        void operator+= (const int8_t& vec) { do_scalar(*this, vec, add); }
        void operator-= (const int8_t& vec) { do_scalar(*this, vec, sub); }
        #endif
        void operator*= (const vector_i8& vec) { do_scalar(*this, vec, mul); }
        void operator/= (const vector_i8& vec) { do_scalar(*this, vec, div); }
        void operator*= (const int8_t& vec) { do_scalar(*this, vec, mul); }
        void operator/= (const int8_t& vec) { do_scalar(*this, vec, div); }

        #ifdef __AVX2__
        void sum(const vector_i8& a, const vector_i8& b) { AVX2_prtn(*this, a, b, add, add); }
        void dif(const vector_i8& a, const vector_i8& b) { AVX2_prtn(*this, a, b, sub, sub); }
        void sum(const vector_i8& a, const int8_t& b) { AVX2_prtn(*this, a, b, add, add); }
        void dif(const vector_i8& a, const int8_t& b) { AVX2_prtn(*this, a, b, sub, sub); }
        void dif(const int8_t& a, const vector_i8& b) { AVX2_prtn(*this, a, b, sub, sub); }
        #elif defined(__AVX__)
        void sum(const vector_i8& a, const vector_i8& b) { AVX_prtn(*this, a, b, add, add); }
        void dif(const vector_i8& a, const vector_i8& b) { AVX_prtn(*this, a, b, sub, sub); }
        void sum(const vector_i8& a, const int8_t& b) { AVX_prtn(*this, a, b, add, add); }
        void dif(const vector_i8& a, const int8_t& b) { AVX_prtn(*this, a, b, sub, sub); }
        void dif(const int8_t& a, const vector_i8& b) { AVX_prtn(*this, a, b, sub, sub); }
        #else
        void sum(const vector_i8& a, const vector_i8& b) { do_scalar(a, b, add); }
        void dif(const vector_i8& a, const vector_i8& b) { do_scalar(a, b, sub); }
        void sum(const vector_i8& a, const int8_t& b) { do_scalar(a, b, add); }
        void dif(const vector_i8& a, const int8_t& b) { do_scalar(a, b, sub); }
        void dif(const int8_t& a, const vector_i8& b) { do_scalar(a, b, sub); }
        #endif
        void pro(const vector_i8& a, const vector_i8& b) { do_scalar(a, b, mul); }
        void quo(const vector_i8& a, const vector_i8& b) { do_scalar(a, b, div); }
        void pro(const vector_i8& a, const int8_t& b) { do_scalar(a, b, mul); }
        void quo(const vector_i8& a, const int8_t& b) { do_scalar(a, b, div); }
        void quo(const int8_t& a, const vector_i8& b) { do_scalar(a, b, div); }

        #ifdef __AVX2__
        vector_i8 operator+ (const vector_i8& vec) { return do_AVX2_create(*this, vec, add, add); }
        vector_i8 operator- (const vector_i8& vec) { return do_AVX2_create(*this, vec, sub, sub); }
        vector_i8 operator+ (const int8_t& x) { return do_AVX2_create(*this, x, add, add); }
        vector_i8 operator- (const int8_t& x) { return do_AVX2_create(*this, x, sub, sub); }
        #elif defined(__AVX__)
        vector_i8 operator+ (const vector_i8& vec) { return do_AVX_create(*this, vec, add, add); }
        vector_i8 operator- (const vector_i8& vec) { return do_AVX_create(*this, vec, sub, sub); }
        vector_i8 operator+ (const int8_t& x) { return do_AVX_create(*this, x, add, add); }
        vector_i8 operator- (const int8_t& x) { return do_AVX_create(*this, x, sub, sub); }
        #else
        vector_i8 operator+ (const vector_i8& vec) { return do_scalar_create(*this, vec, add); }
        vector_i8 operator- (const vector_i8& vec) { return do_scalar_create(*this, vec, sub); }
        vector_i8 operator+ (const int8_t& x) { return do_scalar_create(*this, x, add); }
        vector_i8 operator- (const int8_t& x) { return do_scalar_create(*this, x, sub); }
        #endif

        vector_i8 operator* (const vector_i8& vec) { return do_scalar_create(*this, vec, mul); }
        vector_i8 operator/ (const vector_i8& vec) { return do_scalar_create(*this, vec, div); }
        vector_i8 operator* (const int8_t& x) { return do_scalar_create(*this, x, mul); }
        vector_i8 operator/ (const int8_t& x) { return do_scalar_create(*this, x, div); }
    };
};

#endif