/***************************************************************************
 * Copyright (c) Johan Mabille, Sylvain Corlay, Wolf Vollprecht and         *
 * Martin Renou                                                             *
 * Copyright (c) QuantStack                                                 *
 * Copyright (c) Serge Guelton                                              *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#ifndef XSIMD_AVX2_HPP
#define XSIMD_AVX2_HPP

#include <complex>
#include <type_traits>

#include "../types/xsimd_avx2_register.hpp"

namespace xsimd
{

    namespace kernel
    {
        using namespace types;

        // abs
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> abs(batch<T, A> const& self, requires_arch<avx2>)
        {
            if (std::is_signed<T>::value)
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm256_abs_epi8(self);
                case 2:
                    return _mm256_abs_epi16(self);
                case 4:
                    return _mm256_abs_epi32(self);
                default:
                    return abs(self, avx {});
                }
            }
            return self;
        }

        // add
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> add(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            switch (sizeof(T))
            {
            case 1:
                return _mm256_add_epi8(self, other);
            case 2:
                return _mm256_add_epi16(self, other);
            case 4:
                return _mm256_add_epi32(self, other);
            case 8:
                return _mm256_add_epi64(self, other);
            default:
                return add(self, other, avx {});
            }
        }

        // bitwise_and
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_and(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            return _mm256_and_si256(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> bitwise_and(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<avx2>)
        {
            return _mm256_and_si256(self, other);
        }

        // bitwise_andnot
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_andnot(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            return _mm256_andnot_si256(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> bitwise_andnot(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<avx2>)
        {
            return _mm256_andnot_si256(self, other);
        }

        // bitwise_not
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_not(batch<T, A> const& self, requires_arch<avx2>)
        {
            return _mm256_xor_si256(self, _mm256_set1_epi32(-1));
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> bitwise_not(batch_bool<T, A> const& self, requires_arch<avx2>)
        {
            return _mm256_xor_si256(self, _mm256_set1_epi32(-1));
        }

        // bitwise_lshift
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_lshift(batch<T, A> const& self, int32_t other, requires_arch<avx2>)
        {
            switch (sizeof(T))
            {
            case 2:
                return _mm256_slli_epi16(self, other);
            case 4:
                return _mm256_slli_epi32(self, other);
            case 8:
                return _mm256_slli_epi64(self, other);
            default:
                return bitwise_lshift(self, other, avx {});
            }
        }

        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_lshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            switch (sizeof(T))
            {
            case 4:
                return _mm256_sllv_epi32(self, other);
            case 8:
                return _mm256_sllv_epi64(self, other);
            default:
                return bitwise_lshift(self, other, avx {});
            }
        }

        // bitwise_or
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_or(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            return _mm256_or_si256(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> bitwise_or(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<avx2>)
        {
            return _mm256_or_si256(self, other);
        }

        // bitwise_rshift
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_rshift(batch<T, A> const& self, int32_t other, requires_arch<avx2>)
        {
            if (std::is_signed<T>::value)
            {
                switch (sizeof(T))
                {
                case 1:
                {
                    __m256i sign_mask = _mm256_set1_epi16((0xFF00 >> other) & 0x00FF);
                    __m256i cmp_is_negative = _mm256_cmpgt_epi8(_mm256_setzero_si256(), self);
                    __m256i res = _mm256_srai_epi16(self, other);
                    return _mm256_or_si256(
                        detail::fwd_to_sse([](__m128i s, __m128i o)
                                           { return bitwise_and(batch<T, sse4_2>(s), batch<T, sse4_2>(o), sse4_2 {}); },
                                           sign_mask, cmp_is_negative),
                        _mm256_andnot_si256(sign_mask, res));
                }
                case 2:
                    return _mm256_srai_epi16(self, other);
                case 4:
                    return _mm256_srai_epi32(self, other);
                default:
                    return bitwise_rshift(self, other, avx {});
                }
            }
            else
            {
                switch (sizeof(T))
                {
                case 2:
                    return _mm256_srli_epi16(self, other);
                case 4:
                    return _mm256_srli_epi32(self, other);
                case 8:
                    return _mm256_srli_epi64(self, other);
                default:
                    return bitwise_rshift(self, other, avx {});
                }
            }
        }

        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_rshift(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            if (std::is_signed<T>::value)
            {
                switch (sizeof(T))
                {
                case 4:
                    return _mm256_srav_epi32(self, other);
                default:
                    return bitwise_rshift(self, other, avx {});
                }
            }
            else
            {
                switch (sizeof(T))
                {
                case 4:
                    return _mm256_srlv_epi32(self, other);
                case 8:
                    return _mm256_srlv_epi64(self, other);
                default:
                    return bitwise_rshift(self, other, avx {});
                }
            }
        }

        // bitwise_xor
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_xor(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            return _mm256_xor_si256(self, other);
        }
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> bitwise_xor(batch_bool<T, A> const& self, batch_bool<T, A> const& other, requires_arch<avx2>)
        {
            return _mm256_xor_si256(self, other);
        }

        // complex_low
        template <class A>
        batch<double, A> inline complex_low(batch<std::complex<double>, A> const& self, requires_arch<avx2>)
        {
            __m256d tmp0 = _mm256_permute4x64_pd(self.real(), _MM_SHUFFLE(3, 1, 1, 0));
            __m256d tmp1 = _mm256_permute4x64_pd(self.imag(), _MM_SHUFFLE(1, 2, 0, 0));
            return _mm256_blend_pd(tmp0, tmp1, 10);
        }

        // complex_high
        template <class A>
        batch<double, A> inline complex_high(batch<std::complex<double>, A> const& self, requires_arch<avx2>)
        {
            __m256d tmp0 = _mm256_permute4x64_pd(self.real(), _MM_SHUFFLE(3, 3, 1, 2));
            __m256d tmp1 = _mm256_permute4x64_pd(self.imag(), _MM_SHUFFLE(3, 2, 2, 0));
            return _mm256_blend_pd(tmp0, tmp1, 10);
        }

        // eq
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> eq(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            switch (sizeof(T))
            {
            case 1:
                return _mm256_cmpeq_epi8(self, other);
            case 2:
                return _mm256_cmpeq_epi16(self, other);
            case 4:
                return _mm256_cmpeq_epi32(self, other);
            case 8:
                return _mm256_cmpeq_epi64(self, other);
            default:
                return eq(self, other, avx {});
            }
        }

        // gt
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch_bool<T, A> gt(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            if (std::is_signed<T>::value)
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm256_cmpgt_epi8(self, other);
                case 2:
                    return _mm256_cmpgt_epi16(self, other);
                case 4:
                    return _mm256_cmpgt_epi32(self, other);
                case 8:
                    return _mm256_cmpgt_epi64(self, other);
                default:
                    return gt(self, other, avx {});
                }
            }
            else
            {
                return gt(self, other, avx {});
            }
        }

        // hadd
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline T hadd(batch<T, A> const& self, requires_arch<avx2>)
        {
            switch (sizeof(T))
            {
            case 4:
            {
                __m256i tmp1 = _mm256_hadd_epi32(self, self);
                __m256i tmp2 = _mm256_hadd_epi32(tmp1, tmp1);
                __m128i tmp3 = _mm256_extracti128_si256(tmp2, 1);
                __m128i tmp4 = _mm_add_epi32(_mm256_castsi256_si128(tmp2), tmp3);
                return _mm_cvtsi128_si32(tmp4);
            }
            case 8:
            {
                __m256i tmp1 = _mm256_shuffle_epi32(self, 0x0E);
                __m256i tmp2 = _mm256_add_epi64(self, tmp1);
                __m128i tmp3 = _mm256_extracti128_si256(tmp2, 1);
                __m128i res = _mm_add_epi64(_mm256_castsi256_si128(tmp2), tmp3);
#if defined(__x86_64__)
                return _mm_cvtsi128_si64(res);
#else
                __m128i m;
                _mm_storel_epi64(&m, res);
                int64_t i;
                std::memcpy(&i, &m, sizeof(i));
                return i;
#endif
            }
            default:
                return hadd(self, avx {});
            }
        }
        // load_complex
        template <class A>
        batch<std::complex<float>, A> inline load_complex(batch<float, A> const& hi, batch<float, A> const& lo, requires_arch<avx2>)
        {
            using batch_type = batch<float, A>;
            batch_type real = _mm256_castpd_ps(
                _mm256_permute4x64_pd(
                    _mm256_castps_pd(_mm256_shuffle_ps(hi, lo, _MM_SHUFFLE(2, 0, 2, 0))),
                    _MM_SHUFFLE(3, 1, 2, 0)));
            batch_type imag = _mm256_castpd_ps(
                _mm256_permute4x64_pd(
                    _mm256_castps_pd(_mm256_shuffle_ps(hi, lo, _MM_SHUFFLE(3, 1, 3, 1))),
                    _MM_SHUFFLE(3, 1, 2, 0)));
            return { real, imag };
        }
        template <class A>
        inline batch<std::complex<double>, A> load_complex(batch<double, A> const& hi, batch<double, A> const& lo, requires_arch<avx2>)
        {
            using batch_type = batch<double, A>;
            batch_type real = _mm256_permute4x64_pd(_mm256_unpacklo_pd(hi, lo), _MM_SHUFFLE(3, 1, 2, 0));
            batch_type imag = _mm256_permute4x64_pd(_mm256_unpackhi_pd(hi, lo), _MM_SHUFFLE(3, 1, 2, 0));
            return { real, imag };
        }

        // max
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> max(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            if (std::is_signed<T>::value)
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm256_max_epi8(self, other);
                case 2:
                    return _mm256_max_epi16(self, other);
                case 4:
                    return _mm256_max_epi32(self, other);
                default:
                    return max(self, other, avx {});
                }
            }
            else
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm256_max_epu8(self, other);
                case 2:
                    return _mm256_max_epu16(self, other);
                case 4:
                    return _mm256_max_epu32(self, other);
                default:
                    return max(self, other, avx {});
                }
            }
        }

        // min
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> min(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            if (std::is_signed<T>::value)
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm256_min_epi8(self, other);
                case 2:
                    return _mm256_min_epi16(self, other);
                case 4:
                    return _mm256_min_epi32(self, other);
                default:
                    return min(self, other, avx {});
                }
            }
            else
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm256_min_epu8(self, other);
                case 2:
                    return _mm256_min_epu16(self, other);
                case 4:
                    return _mm256_min_epu32(self, other);
                default:
                    return min(self, other, avx {});
                }
            }
        }

        // mul
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> mul(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            switch (sizeof(T))
            {
            case 2:
                return _mm256_mullo_epi16(self, other);
            case 4:
                return _mm256_mullo_epi32(self, other);
            case 8:
            {
                // from vectorclass, under Apache 2 license
                // https://github.com/vectorclass/version2/blob/a0a33986fb1fe8a5b7844e8a1b1f197ce19af35d/vectori256.h#L3369
                __m256i bswap = _mm256_shuffle_epi32(other, 0xB1); // swap H<->L
                __m256i prodlh = _mm256_mullo_epi32(self, bswap); // 32 bit L*H products
                __m256i zero = _mm256_setzero_si256(); // 0
                __m256i prodlh2 = _mm256_hadd_epi32(prodlh, zero); // a0Lb0H+a0Hb0L,a1Lb1H+a1Hb1L,0,0
                __m256i prodlh3 = _mm256_shuffle_epi32(prodlh2, 0x73); // 0, a0Lb0H+a0Hb0L, 0, a1Lb1H+a1Hb1L
                __m256i prodll = _mm256_mul_epu32(self, other); // a0Lb0L,a1Lb1L, 64 bit unsigned products
                return _mm256_add_epi64(prodll, prodlh3); // a0Lb0L+(a0Lb0H+a0Hb0L)<<32, a1Lb1L+(a1Lb1H+a1Hb1L)<<32
            }
            default:
                return mul(self, other, avx {});
            }
        }

        // sadd
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> sadd(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            if (std::is_signed<T>::value)
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm256_adds_epi8(self, other);
                case 2:
                    return _mm256_adds_epi16(self, other);
                default:
                    return sadd(self, other, avx {});
                }
            }
            else
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm256_adds_epu8(self, other);
                case 2:
                    return _mm256_adds_epu16(self, other);
                default:
                    return sadd(self, other, avx {});
                }
            }
        }

        // select
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> select(batch_bool<T, A> const& cond, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<avx2>)
        {
            switch (sizeof(T))
            {
            case 1:
                return _mm256_blendv_epi8(false_br, true_br, cond);
            case 2:
                return _mm256_blendv_epi8(false_br, true_br, cond);
            case 4:
                return _mm256_blendv_epi8(false_br, true_br, cond);
            case 8:
                return _mm256_blendv_epi8(false_br, true_br, cond);
            default:
                return select(cond, true_br, false_br, avx {});
            }
        }
        template <class A, class T, bool... Values, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> select(batch_bool_constant<batch<T, A>, Values...> const&, batch<T, A> const& true_br, batch<T, A> const& false_br, requires_arch<avx2>)
        {
            constexpr int mask = batch_bool_constant<batch<T, A>, Values...>::mask();
            switch (sizeof(T))
            {
            // FIXME: for some reason mask here is not considered as an immediate,
            // but it's okay for _mm256_blend_epi32
            // case 2: return _mm256_blend_epi16(false_br, true_br, mask);
            case 4:
                return _mm256_blend_epi32(false_br, true_br, mask);
            case 8:
            {
                constexpr int imask = detail::interleave(mask);
                return _mm256_blend_epi32(false_br, true_br, imask);
            }
            default:
                return select(batch_bool<T, A> { Values... }, true_br, false_br, avx2 {});
            }
        }

        // ssub
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> ssub(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            if (std::is_signed<T>::value)
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm256_subs_epi8(self, other);
                case 2:
                    return _mm256_subs_epi16(self, other);
                default:
                    return ssub(self, other, avx {});
                }
            }
            else
            {
                switch (sizeof(T))
                {
                case 1:
                    return _mm256_subs_epu8(self, other);
                case 2:
                    return _mm256_subs_epu16(self, other);
                default:
                    return ssub(self, other, avx {});
                }
            }
        }

        // sub
        template <class A, class T, class = typename std::enable_if<std::is_integral<T>::value, void>::type>
        inline batch<T, A> sub(batch<T, A> const& self, batch<T, A> const& other, requires_arch<avx2>)
        {
            switch (sizeof(T))
            {
            case 1:
                return _mm256_sub_epi8(self, other);
            case 2:
                return _mm256_sub_epi16(self, other);
            case 4:
                return _mm256_sub_epi32(self, other);
            case 8:
                return _mm256_sub_epi64(self, other);
            default:
                return sub(self, other, avx {});
            }
        }

        namespace detail
        {
            template <std::size_t required_scale, std::size_t... scales>
            struct select_scale;

            template <std::size_t required_scale, std::size_t head>
            struct select_scale<required_scale, head>
            {
                static constexpr size_t scale = head;
            };

            template <std::size_t required_scale, std::size_t head, std::size_t... tail>
            struct select_scale<required_scale, head, tail...>
            {
                static constexpr size_t scale = (required_scale % head == 0)
                    ? head
                    : select_scale<required_scale, tail...>::scale;
            };

            template <std::size_t required_scale, std::size_t... scales>
            struct prescaler
            {
                static constexpr std::size_t scale = select_scale<required_scale, scales...>::scale;
                static constexpr std::size_t prescale = required_scale / scale;
                static_assert(required_scale % scale == 0, "found unsuitable scale");

                template <typename T>
                static inline T run(T const& offset)
                {
                    if (prescale == 1)
                        return offset;
                    else
                        return prescale * offset;
                }
            };

            template <std::size_t scale = 1, class A>
            inline batch<float, A> gather_impl(float const* mem, batch<int32_t, A> offset, requires_arch<avx2>)
            {
                return _mm256_i32gather_ps(mem, offset, scale);
            }

            template <std::size_t scale = 1, class A>
            inline batch<double, A> gather_impl(double const* mem, batch<int64_t, A> offset, requires_arch<avx2>)
            {
                return _mm256_i64gather_pd(mem, offset, scale);
            }

            template <std::size_t scale = 1, class A, class T,
                      typename std::enable_if<std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value,
                                              int>::type
                      = 0>
            inline batch<T, A> gather_impl(T const* mem, batch<int32_t, A> offset, requires_arch<avx2>)
            {
                return _mm256_i32gather_epi32((int const*)mem, offset, scale);
            }

            template <std::size_t scale = 1, class A, class T,
                      typename std::enable_if<std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value,
                                              int>::type
                      = 0>
            inline batch<T, A> gather_impl(T const* mem, batch<int64_t, A> offset, requires_arch<avx2>)
            {
                return _mm256_i64gather_epi64((long long int const*)mem, offset, scale);
            }
        };

        template <std::size_t scale, class A, class T, class O,
                  class offset_batch_t = batch<typename O::value_type, A>>
        inline auto gather(T const* mem, O const& offset, requires_arch<avx2>)
            -> decltype(detail::gather_impl(mem, std::declval<offset_batch_t>(), A {}))
        {
            offset_batch_t offset_batch = offset;
            using prescale = detail::prescaler<scale, 8, 4, 2, 1>;
            return detail::gather_impl<prescale::scale>(mem, prescale::run(offset_batch), A {});
        }
    }

}

#endif
