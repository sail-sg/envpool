#ifndef ENVPOOL_CORE_ENVPOOL_H_
#define ENVPOOL_CORE_ENVPOOL_H_

#include "envpool/core/array.h"
#include "envpool/core/dict.h"

/**
 * EnvSpec funciton, it constructs the env spec when a Config is passed.
 */
template <typename EnvFns>
class EnvSpec {
 public:
  typedef decltype(EnvFns::DefaultConfig()) Config;
  typedef typename Config::Keys ConfigKeys;
  typedef typename Config::Values ConfigValues;
  typedef decltype(EnvFns::StateSpec(std::declval<Config>())) StateSpec;
  typedef decltype(EnvFns::ActionSpec(std::declval<Config>())) ActionSpec;
  typedef typename StateSpec::Keys StateKeys;
  typedef typename ActionSpec::Keys ActionKeys;

 public:
  // For C++
  Config config;
  StateSpec state_spec;
  ActionSpec action_spec;
  static inline const Config default_config = EnvFns::DefaultConfig();

 public:
  explicit EnvSpec(const ConfigValues& conf)
      : config(conf),
        state_spec(EnvFns::StateSpec(config)),
        action_spec(EnvFns::ActionSpec(config)) {}
};

/**
 * Templated subclass of EnvPool,
 * to be overrided by the real EnvPool.
 */
template <typename EnvSpec>
class EnvPool {
 protected:
  EnvSpec spec_;

 public:
  typedef EnvSpec Spec;
  typedef Dict<typename EnvSpec::StateKeys, std::vector<Array>> State;
  typedef Dict<typename EnvSpec::ActionKeys, std::vector<Array>> Action;
  explicit EnvPool(const EnvSpec& spec) : spec_(spec) {}

 protected:
  virtual void Send(const std::vector<Array>& action) {
    throw std::runtime_error("send not implmented");
  }
  virtual std::vector<Array> Recv() {
    throw std::runtime_error("recv not implmented");
  }
  virtual void Reset(const Array& env_ids) {
    throw std::runtime_error("reset not implmented");
  }
};

#endif  // ENVPOOL_CORE_ENVPOOL_H_
