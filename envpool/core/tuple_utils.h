#ifndef ENVPOOL_CORE_TUPLE_UTILS_H_
#define ENVPOOL_CORE_TUPLE_UTILS_H_

#include <functional>
#include <tuple>
#include <type_traits>

template <class T, class Tuple>
struct Index;

template <class T, class... Types>
struct Index<T, std::tuple<T, Types...>> {
  static constexpr std::size_t value = 0;
};

template <class T, class U, class... Types>
struct Index<T, std::tuple<U, Types...>> {
  static constexpr std::size_t value =
      1 + Index<T, std::tuple<Types...>>::value;
};

template <class F, class K, class V, std::size_t... I>
decltype(auto) ApplyZip(F &&f, K &&k, V &&v, std::index_sequence<I...>) {
  return std::invoke(std::forward<F>(f),
                     std::make_tuple(I, std::get<I>(std::forward<K>(k)),
                                     std::get<I>(std::forward<V>(v)))...);
}

template <typename... T>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<T>()...));

#endif  // ENVPOOL_CORE_TUPLE_UTILS_H_
