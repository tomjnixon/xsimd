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

#ifndef XSIMD_BATCH_HPP
#define XSIMD_BATCH_HPP

#include <cassert>
#include <complex>

#include "../config/xsimd_arch.hpp"
#include "../memory/xsimd_alignment.hpp"
#include "./xsimd_utils.hpp"

namespace xsimd
{

    /**
     * @brief batch of integer or floating point values.
     *
     * Abstract representation of an SIMD register for floating point or integral
     * value.
     *
     * @tparam T the type of the underlying values.
     * @tparam A the architecture this batch is tied too.
     **/
    template <class T, class A = default_arch>
    class batch : public types::simd_register<T, A>
    {
    public:
        static constexpr std::size_t size = sizeof(types::simd_register<T, A>) / sizeof(T);

        using value_type = T;
        using arch_type = A;
        using register_type = typename types::simd_register<T, A>::register_type;
        using batch_bool_type = batch_bool<T, A>;

        // constructors
        batch() = default;
        batch(T val);
        batch(std::initializer_list<T> data);
        explicit batch(batch_bool_type const& b);
        batch(register_type reg);

        template <class U>
        static XSIMD_NO_DISCARD batch broadcast(U val);

        // memory operators
        template <class U>
        void store_aligned(U* mem) const;
        template <class U>
        void store_unaligned(U* mem) const;
        template <class U>
        void store(U* mem, aligned_mode) const;
        template <class U>
        void store(U* mem, unaligned_mode) const;

        template <class U>
        static XSIMD_NO_DISCARD batch load_aligned(U const* mem);
        template <class U>
        static XSIMD_NO_DISCARD batch load_unaligned(U const* mem);
        template <class U>
        static XSIMD_NO_DISCARD batch load(U const* mem, aligned_mode);
        template <class U>
        static XSIMD_NO_DISCARD batch load(U const* mem, unaligned_mode);

        T get(std::size_t i) const;

        // comparison operators
        batch_bool_type operator==(batch const& other) const;
        batch_bool_type operator!=(batch const& other) const;
        batch_bool_type operator>=(batch const& other) const;
        batch_bool_type operator<=(batch const& other) const;
        batch_bool_type operator>(batch const& other) const;
        batch_bool_type operator<(batch const& other) const;

        // Update operators
        batch& operator+=(batch const& other);
        batch& operator-=(batch const& other);
        batch& operator*=(batch const& other);
        batch& operator/=(batch const& other);
        batch& operator%=(batch const& other);
        batch& operator&=(batch const& other);
        batch& operator|=(batch const& other);
        batch& operator^=(batch const& other);
        batch& operator>>=(int32_t other);
        batch& operator>>=(batch const& other);
        batch& operator<<=(int32_t other);
        batch& operator<<=(batch const& other);

        // incr/decr operators
        batch& operator++();
        batch& operator--();
        batch operator++(int);
        batch operator--(int);

        // unary operators
        batch_bool_type operator!() const;
        batch operator~() const;
        batch operator-() const;
        batch operator+() const;

        // arithmetic operators. They are defined as friend to enable automatic
        // conversion of parameters from scalar to batch. Inline implementation
        // is required to avoid warnings.
        friend batch operator+(batch const& self, batch const& other)
        {
            return batch(self) += other;
        }

        friend batch operator-(batch const& self, batch const& other)
        {
            return batch(self) -= other;
        }

        friend batch operator*(batch const& self, batch const& other)
        {
            return batch(self) *= other;
        }

        friend batch operator/(batch const& self, batch const& other)
        {
            return batch(self) /= other;
        }

        friend batch operator%(batch const& self, batch const& other)
        {
            return batch(self) %= other;
        }

        friend batch operator&(batch const& self, batch const& other)
        {
            return batch(self) &= other;
        }

        friend batch operator|(batch const& self, batch const& other)
        {
            return batch(self) |= other;
        }

        friend batch operator^(batch const& self, batch const& other)
        {
            return batch(self) ^= other;
        }

        friend batch operator>>(batch const& self, batch const& other)
        {
            return batch(self) >>= other;
        }

        friend batch operator<<(batch const& self, batch const& other)
        {
            return batch(self) <<= other;
        }

        friend batch operator>>(batch const& self, int32_t other)
        {
            return batch(self) >>= other;
        }

        friend batch operator<<(batch const& self, int32_t other)
        {
            return batch(self) <<= other;
        }

        friend batch operator&&(batch const& self, batch const& other)
        {
            return batch(self).logical_and(other);
        }

        friend batch operator||(batch const& self, batch const& other)
        {
            return batch(self).logical_or(other);
        }

    private:
        template <size_t... Is>
        batch(T const* data, detail::index_sequence<Is...>);

        batch logical_and(batch const& other) const;
        batch logical_or(batch const& other) const;
    };

    template <class T, class A>
    constexpr std::size_t batch<T, A>::size;

    /**
     * @brief batch of predicate over scalar or complex values.
     *
     * Abstract representation of a predicate over SIMD register for scalar or
     * complex values.
     *
     * @tparam T the type of the predicated values.
     * @tparam A the architecture this batch is tied too.
     **/
    template <class T, class A = default_arch>
    class batch_bool : public types::get_bool_simd_register_t<T, A>
    {
    public:
        static constexpr std::size_t size = sizeof(types::simd_register<T, A>) / sizeof(T);

        using value_type = bool;
        using base_type = types::get_bool_simd_register_t<T, A>;
        using register_type = typename base_type::register_type;
        using batch_type = batch<T, A>;

        // constructors
        batch_bool() = default;
        batch_bool(bool val);
        batch_bool(register_type reg);
        batch_bool(std::initializer_list<bool> data);

        // memory operators
        void store_aligned(bool* mem) const;
        void store_unaligned(bool* mem) const;
        static XSIMD_NO_DISCARD batch_bool load_aligned(bool const* mem);
        static XSIMD_NO_DISCARD batch_bool load_unaligned(bool const* mem);

        bool get(std::size_t i) const;

        // comparison operators
        batch_bool operator==(batch_bool const& other) const;
        batch_bool operator!=(batch_bool const& other) const;

        // logical operators
        batch_bool operator~() const;
        batch_bool operator!() const;
        batch_bool operator&(batch_bool const& other) const;
        batch_bool operator|(batch_bool const& other) const;
        batch_bool operator&&(batch_bool const& other) const;
        batch_bool operator||(batch_bool const& other) const;

    private:
        template <size_t... Is>
        batch_bool(bool const* data, detail::index_sequence<Is...>);

        template <class U, class... V, size_t I, size_t... Is>
        static register_type make_register(detail::index_sequence<I, Is...>, U u, V... v);

        template <class... V>
        static register_type make_register(detail::index_sequence<>, V... v);
    };

    template <class T, class A>
    constexpr std::size_t batch_bool<T, A>::size;

    /**
     * @brief batch of complex values.
     *
     * Abstract representation of an SIMD register for complex values.
     *
     * @tparam T the type of the underlying values.
     * @tparam A the architecture this batch is tied too.
     **/
    template <class T, class A>
    class batch<std::complex<T>, A>
    {
    public:
        using value_type = std::complex<T>;
        using real_batch = batch<T, A>;
        using arch_type = A;
        static constexpr std::size_t size = real_batch::size;
        using batch_bool_type = batch_bool<T, A>;

        // constructors
        batch() = default;
        batch(value_type const& val);
        batch(real_batch const& real, real_batch const& imag);

        batch(real_batch const& real);
        batch(T val);
        batch(std::initializer_list<value_type> data);
        explicit batch(batch_bool_type const& b);

        // memory operators
        static XSIMD_NO_DISCARD batch load_aligned(const T* real_src, const T* imag_src = nullptr);
        static XSIMD_NO_DISCARD batch load_unaligned(const T* real_src, const T* imag_src = nullptr);
        void store_aligned(T* real_dst, T* imag_dst) const;
        void store_unaligned(T* real_dst, T* imag_dst) const;

        static XSIMD_NO_DISCARD batch load_aligned(const value_type* src);
        static XSIMD_NO_DISCARD batch load_unaligned(const value_type* src);
        void store_aligned(value_type* dst) const;
        void store_unaligned(value_type* dst) const;

        template <class U>
        static XSIMD_NO_DISCARD batch load(U const* mem, aligned_mode);
        template <class U>
        static XSIMD_NO_DISCARD batch load(U const* mem, unaligned_mode);
        template <class U>
        void store(U* mem, aligned_mode) const;
        template <class U>
        void store(U* mem, unaligned_mode) const;

        real_batch real() const;
        real_batch imag() const;

        value_type get(std::size_t i) const;

#ifdef XSIMD_ENABLE_XTL_COMPLEX
        // xtl-related methods
        template <bool i3ec>
        batch(xtl::xcomplex<T, T, i3ec> const& val);
        template <bool i3ec>
        batch(std::initializer_list<xtl::xcomplex<T, T, i3ec>> data);

        template <bool i3ec>
        static XSIMD_NO_DISCARD batch load_aligned(const xtl::xcomplex<T, T, i3ec>* src);
        template <bool i3ec>
        static XSIMD_NO_DISCARD batch load_unaligned(const xtl::xcomplex<T, T, i3ec>* src);
        template <bool i3ec>
        void store_aligned(xtl::xcomplex<T, T, i3ec>* dst) const;
        template <bool i3ec>
        void store_unaligned(xtl::xcomplex<T, T, i3ec>* dst) const;
#endif

        // comparison operators
        batch_bool<T, A> operator==(batch const& other) const;
        batch_bool<T, A> operator!=(batch const& other) const;

        // Update operators
        batch& operator+=(batch const& other);
        batch& operator-=(batch const& other);
        batch& operator*=(batch const& other);
        batch& operator/=(batch const& other);

        // incr/decr operators
        batch& operator++();
        batch& operator--();
        batch operator++(int);
        batch operator--(int);

        // unary operators
        batch_bool_type operator!() const;
        batch operator~() const;
        batch operator-() const;
        batch operator+() const;

        // arithmetic operators. They are defined as friend to enable automatic
        // conversion of parameters from scalar to batch
        friend batch operator+(batch const& self, batch const& other)
        {
            return batch(self) += other;
        }

        friend batch operator-(batch const& self, batch const& other)
        {
            return batch(self) -= other;
        }

        friend batch operator*(batch const& self, batch const& other)
        {
            return batch(self) *= other;
        }

        friend batch operator/(batch const& self, batch const& other)
        {
            return batch(self) /= other;
        }

    private:
        real_batch m_real;
        real_batch m_imag;
    };

    template <class T, class A>
    constexpr std::size_t batch<std::complex<T>, A>::size;

}

#include "../arch/xsimd_isa.hpp"
#include "../types/xsimd_batch_constant.hpp"

namespace xsimd
{

    /**********************
     * batch constructors *
     **********************/

    template <class T, class A>
    inline batch<T, A>::batch(T val)
        : types::simd_register<T, A>(kernel::broadcast<A>(val, A {}))
    {
    }

    template <class T, class A>
    inline batch<T, A>::batch(std::initializer_list<T> data)
        : batch(data.begin(), detail::make_index_sequence<size>())
    {
        assert(data.size() == size && "consistent initialization");
    }

    template <class T, class A>
    inline batch<T, A>::batch(batch_bool<T, A> const& b)
        : batch(kernel::from_bool(b, A {}))
    {
    }

    template <class T, class A>
    inline batch<T, A>::batch(register_type reg)
        : types::simd_register<T, A>({ reg })
    {
    }

    template <class T, class A>
    template <size_t... Is>
    inline batch<T, A>::batch(T const* data, detail::index_sequence<Is...>)
        : batch(kernel::set<A>(batch {}, A {}, data[Is]...))
    {
    }

    template <class T, class A>
    template <class U>
    inline XSIMD_NO_DISCARD batch<T, A> batch<T, A>::broadcast(U val)
    {
        return batch(static_cast<T>(val));
    }

    /**************************
     * batch memory operators *
     **************************/

    /**
     * Copy content of this batch to the buffer \c mem. The
     * memory needs to be aligned.
     * @param mem the memory buffer to read
     */
    template <class T, class A>
    template <class U>
    inline void batch<T, A>::store_aligned(U* mem) const
    {
        kernel::store_aligned<A>(mem, *this, A {});
    }

    /**
     * Copy content of this batch to the buffer \c mem. The
     * memory does not need to be aligned.
     * @param mem the memory buffer to write to
     */
    template <class T, class A>
    template <class U>
    inline void batch<T, A>::store_unaligned(U* mem) const
    {
        kernel::store_unaligned<A>(mem, *this, A {});
    }

    template <class T, class A>
    template <class U>
    inline void batch<T, A>::store(U* mem, aligned_mode) const
    {
        return store_aligned(mem);
    }

    template <class T, class A>
    template <class U>
    inline void batch<T, A>::store(U* mem, unaligned_mode) const
    {
        return store_unaligned(mem);
    }

    /**
     * Loading from aligned memory. May involve a conversion if \c U is different
     * from \c T.
     *
     * @param mem the memory buffer to read from.
     * @return a new batch instance.
     */
    template <class T, class A>
    template <class U>
    inline batch<T, A> batch<T, A>::load_aligned(U const* mem)
    {
        return kernel::load_aligned<A>(mem, kernel::convert<T> {}, A {});
    }

    /**
     * Loading from unaligned memory. May involve a conversion if \c U is different
     * from \c T.
     *
     * @param mem the memory buffer to read from.
     * @return a new batch instance.
     */
    template <class T, class A>
    template <class U>
    inline batch<T, A> batch<T, A>::load_unaligned(U const* mem)
    {
        return kernel::load_unaligned<A>(mem, kernel::convert<T> {}, A {});
    }

    template <class T, class A>
    template <class U>
    inline batch<T, A> batch<T, A>::load(U const* mem, aligned_mode)
    {
        return load_aligned(mem);
    }

    template <class T, class A>
    template <class U>
    inline batch<T, A> batch<T, A>::load(U const* mem, unaligned_mode)
    {
        return load_unaligned(mem);
    }

    template <class T, class A>
    inline T batch<T, A>::get(std::size_t i) const
    {
        alignas(A::alignment()) T buffer[size];
        store_aligned(&buffer[0]);
        return buffer[i];
    }

    /******************************
     * batch comparison operators *
     ******************************/

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator==(batch<T, A> const& other) const
    {
        return kernel::eq<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator!=(batch<T, A> const& other) const
    {
        return kernel::neq<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator>=(batch<T, A> const& other) const
    {
        return kernel::ge<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator<=(batch<T, A> const& other) const
    {
        return kernel::le<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator>(batch<T, A> const& other) const
    {
        return kernel::gt<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator<(batch<T, A> const& other) const
    {
        return kernel::lt<A>(*this, other, A {});
    }

    /**************************
     * batch update operators *
     **************************/

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator+=(batch<T, A> const& other)
    {
        return *this = kernel::add<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator-=(batch<T, A> const& other)
    {
        return *this = kernel::sub<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator*=(batch<T, A> const& other)
    {
        return *this = kernel::mul<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator/=(batch<T, A> const& other)
    {
        return *this = kernel::div<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator%=(batch<T, A> const& other)
    {
        return *this = kernel::mod<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator&=(batch<T, A> const& other)
    {
        return *this = kernel::bitwise_and<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator|=(batch<T, A> const& other)
    {
        return *this = kernel::bitwise_or<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator^=(batch<T, A> const& other)
    {
        return *this = kernel::bitwise_xor<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator>>=(batch<T, A> const& other)
    {
        return *this = kernel::bitwise_rshift<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator<<=(batch<T, A> const& other)
    {
        return *this = kernel::bitwise_lshift<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator>>=(int32_t other)
    {
        return *this = kernel::bitwise_rshift<A>(*this, other, A {});
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator<<=(int32_t other)
    {
        return *this = kernel::bitwise_lshift<A>(*this, other, A {});
    }

    /*****************************
     * batch incr/decr operators *
     *****************************/

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator++()
    {
        return operator+=(1);
    }

    template <class T, class A>
    inline batch<T, A>& batch<T, A>::operator--()
    {
        return operator-=(1);
    }

    template <class T, class A>
    inline batch<T, A> batch<T, A>::operator++(int)
    {
        batch<T, A> copy(*this);
        operator+=(1);
        return copy;
    }

    template <class T, class A>
    inline batch<T, A> batch<T, A>::operator--(int)
    {
        batch copy(*this);
        operator-=(1);
        return copy;
    }

    /*************************
     * batch unary operators *
     *************************/

    template <class T, class A>
    inline batch_bool<T, A> batch<T, A>::operator!() const
    {
        return kernel::eq<A>(*this, batch(0), A {});
    }

    template <class T, class A>
    inline batch<T, A> batch<T, A>::operator~() const
    {
        return kernel::bitwise_not<A>(*this, A {});
    }

    template <class T, class A>
    inline batch<T, A> batch<T, A>::operator-() const
    {
        return kernel::neg<A>(*this, A {});
    }

    template <class T, class A>
    inline batch<T, A> batch<T, A>::operator+() const
    {
        return *this;
    }

    /************************
     * batch private method *
     ************************/

    template <class T, class A>
    inline batch<T, A> batch<T, A>::logical_and(batch<T, A> const& other) const
    {
        return kernel::logical_and<A>(*this, other, A());
    }

    template <class T, class A>
    inline batch<T, A> batch<T, A>::logical_or(batch<T, A> const& other) const
    {
        return kernel::logical_or<A>(*this, other, A());
    }

    /***************************
     * batch_bool constructors *
     ***************************/

    template <class T, class A>
    template <size_t... Is>
    inline batch_bool<T, A>::batch_bool(bool const* data, detail::index_sequence<Is...>)
        : batch_bool(kernel::set<A>(batch_bool {}, A {}, data[Is]...))
    {
    }

    template <class T, class A>
    inline batch_bool<T, A>::batch_bool(register_type reg)
        : types::get_bool_simd_register_t<T, A>({ reg })
    {
    }

    template <class T, class A>
    inline batch_bool<T, A>::batch_bool(std::initializer_list<bool> data)
        : batch_bool(data.begin(), detail::make_index_sequence<size>())
    {
    }

    /*******************************
     * batch_bool memory operators *
     *******************************/

    template <class T, class A>
    inline void batch_bool<T, A>::store_aligned(bool* mem) const
    {
        kernel::store(*this, mem, A {});
    }

    template <class T, class A>
    inline void batch_bool<T, A>::store_unaligned(bool* mem) const
    {
        store_aligned(mem);
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::load_aligned(bool const* mem)
    {
        batch_type ref(0);
        alignas(A::alignment()) T buffer[size];
        for (std::size_t i = 0; i < size; ++i)
            buffer[i] = mem[i] ? 1 : 0;
        return ref != batch_type::load_aligned(&buffer[0]);
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::load_unaligned(bool const* mem)
    {
        return load_aligned(mem);
    }

    template <class T, class A>
    inline bool batch_bool<T, A>::get(std::size_t i) const
    {
        alignas(A::alignment()) bool buffer[size];
        store_aligned(&buffer[0]);
        return buffer[i];
    }

    /***********************************
     * batch_bool comparison operators *
     ***********************************/

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator==(batch_bool<T, A> const& other) const
    {
        return kernel::eq<A>(*this, other, A {}).data;
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator!=(batch_bool<T, A> const& other) const
    {
        return kernel::neq<A>(*this, other, A {}).data;
    }

    /********************************
     * batch_bool logical operators *
     ********************************/

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator~() const
    {
        return kernel::bitwise_not<A>(*this, A {}).data;
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator!() const
    {
        return operator==(batch_bool(false));
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator&(batch_bool<T, A> const& other) const
    {
        return kernel::bitwise_and<A>(*this, other, A {}).data;
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator|(batch_bool<T, A> const& other) const
    {
        return kernel::bitwise_or<A>(*this, other, A {}).data;
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator&&(batch_bool const& other) const
    {
        return operator&(other);
    }

    template <class T, class A>
    inline batch_bool<T, A> batch_bool<T, A>::operator||(batch_bool const& other) const
    {
        return operator|(other);
    }

    /******************************
     * batch_bool private methods *
     ******************************/

    template <class T, class A>
    inline batch_bool<T, A>::batch_bool(bool val)
        : base_type { make_register(detail::make_index_sequence<size - 1>(), val) }
    {
    }

    template <class T, class A>
    template <class U, class... V, size_t I, size_t... Is>
    inline auto batch_bool<T, A>::make_register(detail::index_sequence<I, Is...>, U u, V... v) -> register_type
    {
        return make_register(detail::index_sequence<Is...>(), u, u, v...);
    }

    template <class T, class A>
    template <class... V>
    inline auto batch_bool<T, A>::make_register(detail::index_sequence<>, V... v) -> register_type
    {
        return kernel::set<A>(batch_bool<T, A>(), A {}, v...).data;
    }

    /*******************************
     * batch<complex> constructors *
     *******************************/

    template <class T, class A>
    inline batch<std::complex<T>, A>::batch(value_type const& val)
        : m_real(val.real())
        , m_imag(val.imag())
    {
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>::batch(real_batch const& real, real_batch const& imag)
        : m_real(real)
        , m_imag(imag)
    {
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>::batch(real_batch const& real)
        : m_real(real)
        , m_imag(0)
    {
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>::batch(T val)
        : m_real(val)
        , m_imag(0)
    {
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>::batch(std::initializer_list<value_type> data)
    {
        *this = load_unaligned(data.begin());
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>::batch(batch_bool_type const& b)
        : m_real(b)
        , m_imag(0)
    {
    }

    /***********************************
     * batch<complex> memory operators *
     ***********************************/

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load_aligned(const T* real_src, const T* imag_src)
    {
        return { batch<T, A>::load_aligned(real_src), imag_src ? batch<T, A>::load_aligned(imag_src) : batch<T, A>(0) };
    }
    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load_unaligned(const T* real_src, const T* imag_src)
    {
        return { batch<T, A>::load_unaligned(real_src), imag_src ? batch<T, A>::load_unaligned(imag_src) : batch<T, A>(0) };
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load_aligned(const value_type* src)
    {
        return kernel::load_complex_aligned<A>(src, kernel::convert<value_type> {}, A {});
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load_unaligned(const value_type* src)
    {
        return kernel::load_complex_unaligned<A>(src, kernel::convert<value_type> {}, A {});
    }

    template <class T, class A>
    inline void batch<std::complex<T>, A>::store_aligned(value_type* dst) const
    {
        return kernel::store_complex_aligned(dst, *this, A {});
    }

    template <class T, class A>
    inline void batch<std::complex<T>, A>::store_unaligned(value_type* dst) const
    {
        return kernel::store_complex_unaligned(dst, *this, A {});
    }

    template <class T, class A>
    inline void batch<std::complex<T>, A>::store_aligned(T* real_dst, T* imag_dst) const
    {
        m_real.store_aligned(real_dst);
        m_imag.store_aligned(imag_dst);
    }

    template <class T, class A>
    inline void batch<std::complex<T>, A>::store_unaligned(T* real_dst, T* imag_dst) const
    {
        m_real.store_unaligned(real_dst);
        m_imag.store_unaligned(imag_dst);
    }

    template <class T, class A>
    template <class U>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load(U const* mem, aligned_mode)
    {
        return load_aligned(mem);
    }

    template <class T, class A>
    template <class U>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load(U const* mem, unaligned_mode)
    {
        return load_unaligned(mem);
    }

    template <class T, class A>
    template <class U>
    inline void batch<std::complex<T>, A>::store(U* mem, aligned_mode) const
    {
        return store_aligned(mem);
    }

    template <class T, class A>
    template <class U>
    inline void batch<std::complex<T>, A>::store(U* mem, unaligned_mode) const
    {
        return store_unaligned(mem);
    }

    template <class T, class A>
    inline auto batch<std::complex<T>, A>::real() const -> real_batch
    {
        return m_real;
    }

    template <class T, class A>
    inline auto batch<std::complex<T>, A>::imag() const -> real_batch
    {
        return m_imag;
    }

    template <class T, class A>
    inline auto batch<std::complex<T>, A>::get(std::size_t i) const -> value_type
    {
        alignas(A::alignment()) value_type buffer[size];
        store_aligned(&buffer[0]);
        return buffer[i];
    }

    /**************************************
     * batch<complex> xtl-related methods *
     **************************************/

#ifdef XSIMD_ENABLE_XTL_COMPLEX

    template <class T, class A>
    template <bool i3ec>
    inline batch<std::complex<T>, A>::batch(xtl::xcomplex<T, T, i3ec> const& val)
        : m_real(val.real())
        , m_imag(val.imag())
    {
    }

    template <class T, class A>
    template <bool i3ec>
    inline batch<std::complex<T>, A>::batch(std::initializer_list<xtl::xcomplex<T, T, i3ec>> data)
    {
        *this = load_unaligned(data.begin());
    }

    // Memory layout of an xcomplex and std::complex are the same when xcomplex
    // stores values and not reference. Unfortunately, this breaks strict
    // aliasing...

    template <class T, class A>
    template <bool i3ec>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load_aligned(const xtl::xcomplex<T, T, i3ec>* src)
    {
        return load_aligned(reinterpret_cast<std::complex<T> const*>(src));
    }

    template <class T, class A>
    template <bool i3ec>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::load_unaligned(const xtl::xcomplex<T, T, i3ec>* src)
    {
        return load_unaligned(reinterpret_cast<std::complex<T> const*>(src));
    }

    template <class T, class A>
    template <bool i3ec>
    inline void batch<std::complex<T>, A>::store_aligned(xtl::xcomplex<T, T, i3ec>* dst) const
    {
        store_aligned(reinterpret_cast<std::complex<T>*>(dst));
    }

    template <class T, class A>
    template <bool i3ec>
    inline void batch<std::complex<T>, A>::store_unaligned(xtl::xcomplex<T, T, i3ec>* dst) const
    {
        store_unaligned(reinterpret_cast<std::complex<T>*>(dst));
    }

#endif

    /***************************************
     * batch<complex> comparison operators *
     ***************************************/

    template <class T, class A>
    inline batch_bool<T, A> batch<std::complex<T>, A>::operator==(batch const& other) const
    {
        return m_real == other.m_real && m_imag == other.m_imag;
    }

    template <class T, class A>
    inline batch_bool<T, A> batch<std::complex<T>, A>::operator!=(batch const& other) const
    {
        return m_real != other.m_real || m_imag != other.m_imag;
    }

    /***********************************
     * batch<complex> update operators *
     ***********************************/

    template <class T, class A>
    inline batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator+=(batch const& other)
    {
        m_real += other.m_real;
        m_imag += other.m_imag;
        return *this;
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator-=(batch const& other)
    {
        m_real -= other.m_real;
        m_imag -= other.m_imag;
        return *this;
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator*=(batch const& other)
    {
        real_batch new_real = real() * other.real() - imag() * other.imag();
        real_batch new_imag = real() * other.imag() + imag() * other.real();
        m_real = new_real;
        m_imag = new_imag;
        return *this;
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator/=(batch const& other)
    {
        real_batch a = real();
        real_batch b = imag();
        real_batch c = other.real();
        real_batch d = other.imag();
        real_batch e = c * c + d * d;
        m_real = (c * a + d * b) / e;
        m_imag = (c * b - d * a) / e;
        return *this;
    }

    /**************************************
     * batch<complex> incr/decr operators *
     **************************************/

    template <class T, class A>
    inline batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator++()
    {
        return operator+=(1);
    }

    template <class T, class A>
    inline batch<std::complex<T>, A>& batch<std::complex<T>, A>::operator--()
    {
        return operator-=(1);
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::operator++(int)
    {
        batch copy(*this);
        operator+=(1);
        return copy;
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::operator--(int)
    {
        batch copy(*this);
        operator-=(1);
        return copy;
    }

    /**********************************
     * batch<complex> unary operators *
     **********************************/

    template <class T, class A>
    inline batch_bool<T, A> batch<std::complex<T>, A>::operator!() const
    {
        return operator==(batch(0));
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::operator~() const
    {
        return { ~m_real, ~m_imag };
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::operator-() const
    {
        return { -m_real, -m_imag };
    }

    template <class T, class A>
    inline batch<std::complex<T>, A> batch<std::complex<T>, A>::operator+() const
    {
        return { +m_real, +m_imag };
    }
}

#endif
