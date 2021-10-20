#ifndef ENVPOOL_CORE_DICT_H_
#define ENVPOOL_CORE_DICT_H_

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "envpool/core/array.h"
#include "envpool/core/spec.h"
#include "envpool/core/tuple_utils.h"
#include "envpool/core/type_utils.h"

template <typename K, typename D>
class Value {
 public:
  using Key = K;
  using Type = D;
  explicit Value(Type &&v) : v(v) {}
  Type v;
};

template <char... C>
class Key {
 public:
  static constexpr const inline char _str[sizeof...(C) + 1]{C..., '\0'};
  static constexpr const inline std::string_view str_view{_str, sizeof...(C)};
  template <typename Type>
  static constexpr inline auto bind(Type &&v) {
    return Value<Key, Type>(std::forward<Type>(v));
  }
  static inline std::string str() { return {str_view.data(), str_view.size()}; }
};

template <class CharT, CharT... CS>
inline constexpr auto operator""_() {
  return Key<CS...>{};
}

template <
    typename Key, typename Keys, typename TupleOrVector,
    std::enable_if_t<is_tuple_v<std::decay_t<TupleOrVector>>, bool> = true>
inline decltype(auto) Take(const Key &key, TupleOrVector &&values) {
  constexpr std::size_t index = Index<Key, Keys>::value;
  return std::get<index>(std::forward<TupleOrVector>(values));
}

template <
    typename Key, typename Keys, typename TupleOrVector,
    std::enable_if_t<is_vector_v<std::decay_t<TupleOrVector>>, bool> = true>
inline decltype(auto) Take(const Key &key, TupleOrVector &&values) {
  constexpr std::size_t index = Index<Key, Keys>::value;
  return std::forward<TupleOrVector>(values).at(index);
}

template <typename StringKeys, typename TupleOrVector,
          typename = std::enable_if_t<is_tuple_v<StringKeys>>>
class Dict : public std::decay_t<TupleOrVector> {
 public:
  typedef std::decay_t<TupleOrVector> Values;
  typedef StringKeys Keys;
  constexpr static std::size_t size = std::tuple_size<Keys>::value;

  /**
   * Check that the size of values / keys tuple should match
   */
  template <typename V = Values, std::enable_if_t<is_tuple_v<V>, bool> = true>
  void Check() {
    static_assert(std::tuple_size_v<Keys> == std::tuple_size_v<Values>,
                  "Number of keys and values doesn't match");
  }

  template <typename V = Values, std::enable_if_t<is_vector_v<V>, bool> = true>
  void Check() {
    DCHECK_EQ(std::tuple_size<Keys>(), static_cast<V *>(this)->size())
        << "Size must match";
  }

  /**
   * Constructor, makes a dict from keys and values
   */
  Dict(const Keys &keys, TupleOrVector &&values) : Values(std::move(values)) {
    Check();
  }

  Dict(const Keys &keys, const TupleOrVector &values) : Values(values) {
    Check();
  }

  /**
   * Constructor, needs to be called with template types
   */
  explicit Dict(TupleOrVector &&values) : Values(std::move(values)) { Check(); }

  explicit Dict(const TupleOrVector &values) : Values(values) { Check(); }

  /**
   * Gives the values a [index] based accessor
   * converts the string literal to a compile time index, and use
   * std::get<index> to get it from the base class.
   * If the key doesn't exists in the keys, compilation will fail.
   */
  template <typename Key,
            std::enable_if_t<any_match<Key, Keys>::value, bool> = true>
  inline decltype(auto) operator[](const Key &key) {
    return Take<Key, Keys, Values &>(key, *this);
  }
  template <typename Key,
            std::enable_if_t<any_match<Key, Keys>::value, bool> = true>
  inline decltype(auto) operator[](const Key &key) const {
    return Take<Key, Keys, const Values &>(key, *this);
  }

  /**
   * Return a static constexpr list of all the keys in a tuple.
   */
  static constexpr decltype(auto) static_keys() { return Keys(); }

  /**
   * Return a list of all the keys as strings.
   */
  static std::vector<std::string> keys() {
    std::vector<std::string> rets;
    std::apply([&](auto &&...key) { (rets.push_back(key.str()), ...); },
               Keys());
    return rets;
  }

  /**
   * Return a static list of all the values in a tuple.
   */
  Values &values() { return *this; }

  /**
   * Const version of static_values
   */
  const Values &values() const { return *this; }

  /**
   * Convert the value tuple to a dynamic vector of values.
   * This function is only enabled when Values is instantiation of std::tuple,
   * and when all elements in the values can be converted to Type
   */
  template <typename Type, bool IsTuple = is_tuple_v<Values>,
            std::enable_if_t<IsTuple, bool> = true,
            std::enable_if_t<all_convertible<Type, Values>::value, bool> = true>
  std::vector<Type> values() const {
    std::vector<Type> rets;
    std::apply([&](auto &&...value) { (rets.push_back(Type(value)), ...); },
               *static_cast<const Values *>(this));
    return rets;
  }

  /**
   * Convert the value vector to a vector of type `Type`.
   * This function is only enabled when Values is an instantiation of
   * std::vector.
   */
  template <typename Type, bool IsTuple = is_tuple_v<Values>,
            std::enable_if_t<!IsTuple, bool> = true>
  std::vector<Type> values() const {
    return std::vector<Type>(this->begin(), this->end());
  }

  template <class F, bool IsTuple = is_tuple_v<Values>,
            std::enable_if_t<IsTuple, bool> = true>
  decltype(auto) apply(F &&f) const {
    ApplyZip(f, Keys(), *this, std::make_index_sequence<size>{});
  }
};

/**
 * Make a dict which is actually an namedtuple in cpp
 * Syntax is like
 * auto d = MakeDict("abc"_.bind(0.), "xyz"_.bind(0.), "ijk"_.bind(1));
 * The above makes a dict { "abc": 0., "xyz": 0., "ijk": 1 }
 */
template <typename... Value>
decltype(auto) MakeDict(Value... v) {
  return Dict(std::make_tuple(typename Value::Key()...),
              std::make_tuple(v.v...));
}

template <
    typename DictA, typename DictB,
    typename AllKeys = tuple_cat_t<typename DictA::Keys, typename DictB::Keys>,
    std::enable_if_t<is_tuple_v<typename DictA::Values> &&
                         is_tuple_v<typename DictB::Values>,
                     bool> = true>
decltype(auto) ConcatDict(const DictA &a, const DictB &b) {
  auto c = std::tuple_cat(static_cast<const typename DictA::Values &>(a),
                          static_cast<const typename DictB::Values &>(b));
  return Dict<AllKeys, decltype(c)>(std::move(c));
}

template <
    typename DictA, typename DictB,
    typename AllKeys = tuple_cat_t<typename DictA::Keys, typename DictB::Keys>,
    std::enable_if_t<is_vector_v<DictA> && is_vector_v<DictB>, bool> = true,
    std::enable_if_t<
        std::is_same_v<typename DictA::Values, typename DictA::Values>, bool> =
        true>
decltype(auto) ConcatDict(const DictA &a, const DictB &b) {
  typedef typename DictA::Values::value_type value_type;
  std::vector<value_type> c;
  c.insert(c.end(), a.begin(), a.end());
  c.insert(c.end(), b.begin(), b.end());
  return Dict<AllKeys, decltype(c)>(c);
}

/**
 * Transform an input vector into an output vector.
 * calls std::transform, infer the vector that needs to be created from the
 * transform function.
 */
template <typename S, typename F,
          typename R = decltype(std::declval<F>()(std::declval<S>()))>
std::vector<R> Transform(const std::vector<S> &src, F &&transform) {
  std::vector<R> tgt;
  std::transform(src.begin(), src.end(), std::back_inserter(tgt),
                 std::forward<F>(transform));
  return tgt;
}

/**
 * Static version of MakeArray.
 * Takes a tuple of `template <typename T> Spec<T>`.
 */
template <typename... Spec>
std::vector<Array> MakeArray(const std::tuple<Spec...> &specs) {
  std::vector<Array> rets;
  std::apply([&](auto &&...spec) { (rets.push_back(Array(spec)), ...); },
             specs);
  return rets;
}

/**
 * Dynamic version of MakeArray.
 * Takes a vector of `ShapeSpec`.
 */
std::vector<Array> MakeArray(const std::vector<ShapeSpec> &specs) {
  return std::vector<Array>(specs.begin(), specs.end());
}

#endif  // ENVPOOL_CORE_DICT_H_
