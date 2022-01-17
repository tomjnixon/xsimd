#ifndef XSIMD_GATHER_UTILS_HPP
#define XSIMD_GATHER_UTILS_HPP

#include "./xsimd_batch.hpp"
#include "./xsimd_utils.hpp"

namespace xsimd
{
    namespace detail
    {
        /// ramp generator for use with make_batch_constant
        struct ramp_generator
        {
            static constexpr size_t get(size_t index, size_t /*size*/)
            {
                return index;
            }
        };
    }

    /// batch-alike whose element i is i * stride
    ///
    /// this is convertible to a batch by multiplying a ramp by the stride,
    /// but also has a get method which calculates the offset directly, to
    /// make fallback code more efficient
    template <typename batch_type_>
    class stride_offset
    {
    public:
        explicit stride_offset(std::size_t stride)
            : stride(stride)
        {
        }

    private:
        using batch_type = batch_type_;

    public:
        using value_type = typename batch_type::value_type;
        static constexpr std::size_t size = batch_type::size;

        constexpr std::size_t get(std::size_t i) const
        {
            return stride * i;
        }

        operator batch_type() const
        {
            constexpr auto ramp = xsimd::make_batch_constant<batch_type, detail::ramp_generator>();
            return (batch_type)ramp * (value_type)stride;
        }

    private:
        std::size_t stride;
    };

    /// batch_bool-alike where the first n elements are true and the rest
    /// are false
    template <class batch_type_>
    class first_n_true
    {
    public:
        explicit constexpr first_n_true(size_t n)
            : n(n)
        {
        }

        using batch_type = batch_type_;
        using value_type = bool;
        using arch_type = typename batch_type::arch_type;
        static constexpr std::size_t size = batch_type::size;

        using batch_bool_type = typename batch_type::batch_bool_type;

        inline operator batch_bool_type() const
        {
            // do the comparison using signed integers, as this is fast on more architectures
            using int_batch = as_integer_t<batch_type>;
            constexpr auto ramp = xsimd::make_batch_constant<int_batch, detail::ramp_generator>();

            return batch_bool_type((int_batch)ramp < int_batch(n));
        }

        inline constexpr bool get(size_t i) const
        {
            return i < n;
        }

    private:
        size_t n;
    };

    namespace detail
    {
        /// batch-alike which is convertible to a signed batch, but has an
        /// get method which returns unsigned values
        ///
        /// use this for values which are representable in either an unsigned
        /// or signed integer, to get the best performance in situations which
        /// have fast SIMD implementations for signed types only but a fallback
        /// implementation which is more efficient with unsigned types
        /// (generally by avoiding explicit sign extension)
        template <typename batch_type_>
        struct allow_signed_conversion
        {
            using batch_type = batch_type_;
            batch_type b;

            using batch_value_type = typename batch_type::value_type;

            using unsigned_t = typename std::make_unsigned<batch_value_type>::type;
            using signed_t = typename std::make_signed<batch_value_type>::type;
            using signed_batch = xsimd::batch<signed_t, typename batch_type::arch_type>;

            // use as batch, pretend to be signed
            static constexpr size_t size = batch_type::size;
            using value_type = signed_t;
            operator signed_batch() const
            {
                return xsimd::bitwise_cast<signed_batch>(b);
            }

            // use as fallback, pretend to be unsigned
            constexpr unsigned_t get(std::size_t i) const
            {
                return static_cast<unsigned_t>(b.get(i));
            }
        };
    }

    template <typename batch_type>
    detail::allow_signed_conversion<batch_type> allow_signed_conversion(batch_type b)
    {
        return { b };
    }
}

#endif
