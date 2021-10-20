#ifndef ENVPOOL_CORE_TYPE_UTILS_H_
#define ENVPOOL_CORE_TYPE_UTILS_H_

#include <functional>
#include <tuple>
#include <type_traits>

template <class T, class TupleTs>
struct any_match;

template <class T, template <typename...> class Tuple, typename... Ts>
struct any_match<T, Tuple<Ts...>> : std::disjunction<std::is_same<T, Ts>...> {};

template <class T, class TupleTs>
struct all_match;

template <class T, template <typename...> class Tuple, typename... Ts>
struct all_match<T, Tuple<Ts...>> : std::conjunction<std::is_same<T, Ts>...> {};

template <class To, class TupleTs>
struct all_convertible;

template <class To, template <typename...> class Tuple, typename... Fs>
struct all_convertible<To, Tuple<Fs...>>
    : std::conjunction<std::is_convertible<Fs, To>...> {};

template <typename T>
constexpr bool is_tuple_v = false;
template <typename... types>
constexpr bool is_tuple_v<std::tuple<types...>> = true;

template <typename T>
constexpr bool is_vector_v = false;
template <typename VT>
constexpr bool is_vector_v<std::vector<VT>> = true;

#endif  // ENVPOOL_CORE_TYPE_UTILS_H_
