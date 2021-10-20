"""Api wrapper layer for EnvPool
"""
from abc import ABC, ABCMeta
from collections import OrderedDict, namedtuple
from typing import (
  Any,
  Dict,
  List,
  NamedTuple,
  Protocol,
  Tuple,
  Type,
  Union,
  no_type_check,
)

import dm_env
import numpy as np
import tree
from dm_env import TimeStep


def check_key_duplication(cls: Any, keytype: str, keys: List[str]) -> None:
  ukeys, counts = np.unique(keys, return_counts=True)
  if not np.all(counts == 1):
    dup_keys = ukeys[counts > 1]
    raise SystemError(
      f"""{cls} c++ code error.
{keytype} keys {list(dup_keys)} are duplicated.
Please report to the author of {cls}.
"""
    )


def check_key_existence(
  cls: Any, keytype: str, keys: List[str], ks: List[str]
) -> None:
  for k in ks:
    if k not in keys:
      raise SystemError(
        f"""{cls} c++ code error.
  {k} is a required {keytype} key, {keys}
  Please report to the author of {cls}.
  """
      )


class EnvSpec(Protocol):
  _config_keys: List[str]
  _default_config_values: Tuple
  config_type: Type

  def __init__(self, config: Tuple):
    """protocol for constructor"""

  @property
  def _state_spec(self) -> List:
    """cpp private _state_spec"""

  @property
  def _action_spec(self) -> List:
    """cpp private _action_spec"""

  @property
  def _config_values(self) -> Tuple:
    """cpp private _config_values"""

  @property
  def config(self) -> NamedTuple:
    """Configuration used to create the current EnvSpec."""

  @property
  def state_spec(self) -> OrderedDict:
    """Specs of the states of the environment."""

  @property
  def action_spec(self) -> OrderedDict:
    """Specs of the actions of the environment."""


class EnvSpecMixin(ABC):
  config_type: Type

  @property
  def config(self: EnvSpec) -> NamedTuple:
    """Configuration used to create the current EnvSpec."""
    return self.config_type(*self._config_values)

  @property
  def state_spec(self: EnvSpec) -> OrderedDict:
    """Specs of the states of the environment.
    Returns:
      state_spec: An ordered dict whose keys are the names of the states,
        its values is a tuple of (dtype, shape).
    """
    return OrderedDict(self._state_spec)

  @property
  def action_spec(self: EnvSpec) -> OrderedDict:
    """Specs of the actions of the environment.
    Returns:
      state_spec: An ordered dict whose keys are the names of the actions,
        its values is a tuple of (dtype, shape).
    """
    return OrderedDict(self._action_spec)


class EnvSpecMeta(ABCMeta):

  def __new__(cls: Any, name: str, parents: Tuple, attrs: Dict) -> Any:
    base = parents[0]
    parents = (base, EnvSpecMixin)
    config_keys = base._config_keys
    check_key_duplication(name, "config", config_keys)
    check_key_existence(name, "config", config_keys, ["num_envs"])
    config_keys: List[str] = list(
      map(lambda s: s.replace(".", "_"), config_keys)
    )
    defaults: Tuple = base._default_config_values
    EnvSpecMixin.config_type = namedtuple(  # type: ignore
        "Config",
        config_keys,
        defaults=defaults,  # type: ignore
    )
    return super().__new__(cls, name, parents, attrs)


class EnvPool(Protocol):
  _state_keys: List[str]
  _action_keys: List[str]

  def __init__(self, spec: EnvSpec):
    """constructor of EnvPool"""

  @property
  def _spec(self) -> EnvSpec:
    """cpp spec"""

  @property
  def _action_spec(self) -> List:
    """cpp spec"""

  def _check_action(self, actions: List) -> None:
    """check action shapes"""

  def _recv(self) -> List[np.ndarray]:
    """cpp private _recv"""

  def _send(self, action: List[np.ndarray]) -> None:
    """cpp private _send"""

  def _reset(self, env_ids: np.ndarray) -> None:
    """cpp private _reset"""

  def _to_dm(self, state: List[np.ndarray]) -> TimeStep:
    """convert to dm-env format"""

  def send(self, action: Union[OrderedDict, List]) -> None:
    """send wrapper"""

  def recv(self) -> TimeStep:
    """recv wrapper"""

  def step(self, action: Union[OrderedDict, List]) -> TimeStep:
    """step interface that performs send/recv"""

  def reset(self) -> TimeStep:
    """reset interface"""


class EnvPoolMixin(ABC):

  def _check_action(self: EnvPool, actions: List) -> None:
    if not hasattr(self, "_action_spec"):
      self._action_spec = self._spec._action_spec
    for a, (k, (_, shape)) in zip(actions, self._action_spec):
      shape = tuple(shape)
      if len(shape) > 0 and shape[0] == -1:
        if a.shape[1:] != shape[1:]:
          raise RuntimeError(f"Expected shape {shape} for {k}, got {a.shape}")
      else:
        if len(a.shape) == 0 or a.shape[1:] != shape:
          raise RuntimeError(
            f"Expected shape {('num_env', *shape)} for {k}, got {a.shape}"
          )

  def send(self: EnvPool, action: Union[OrderedDict, List]) -> None:
    if isinstance(action, OrderedDict):
      action = list(action.values())
    self._check_action(action)
    self._send(action)

  def recv(self: EnvPool) -> TimeStep:
    state_list = self._recv()
    return self._to_dm(state_list)

  def step(self: EnvPool, action: Union[OrderedDict, List]) -> TimeStep:
    self.send(action)
    return self.recv()

  def reset(self: EnvPool, env_ids: np.ndarray) -> None:
    """Follows the async semantics, reset the envs in env_ids."""
    self._reset(env_ids)


def tree_structure(root_name: str, keys: List[str]) -> Tuple[Tuple, List[int]]:
  odict_tree: OrderedDict = OrderedDict()
  for i, key in enumerate(keys):
    segments = key.split(".")
    tr = odict_tree
    for j, s in enumerate(segments):
      if j == len(segments) - 1:
        tr[s] = i
      else:
        if s not in tr:
          tr[s] = OrderedDict()
        tr = tr[s]

  def _to_namedtuple(name: str, odict: OrderedDict) -> Tuple:
    return namedtuple(name, odict.keys())(
      *[
        _to_namedtuple(k, v) if isinstance(v, OrderedDict) else v
        for k, v in odict.items()
      ]
    )

  structure = _to_namedtuple(root_name, odict_tree)
  indice = tree.flatten(structure)
  return structure, indice


class EnvPoolMeta(ABCMeta):

  def __new__(cls: Any, name: str, parents: Tuple, attrs: Dict) -> Any:
    base = parents[0]
    parents = (base, EnvPoolMixin)
    state_keys = base._state_keys
    action_keys = base._action_keys
    check_key_duplication(name, "state", state_keys)
    check_key_duplication(name, "action", action_keys)
    check_key_existence(name, "state", state_keys, ["elapsed_steps", "done"])
    has_reward = ("reward" in state_keys)

    state_structure, state_idx = tree_structure("State", state_keys)
    action_structure, action_idx = tree_structure("Action", action_keys)

    @no_type_check  # because namedtuple is a dynamically generated type
    def _to_dm(self: Any, state_values: List[np.ndarray]) -> TimeStep:
      state = tree.unflatten_as(
        state_structure, [state_values[i] for i in state_idx]
      )
      done = state.done
      elapse = state.elapsed_steps
      step_type = np.full(done.shape, dm_env.StepType.MID)
      step_type[(elapse == 1)] = dm_env.StepType.FIRST
      step_type[done] = dm_env.StepType.LAST
      timestep = TimeStep(
        step_type=step_type,
        observation=state,
        reward=state.reward if has_reward else None,
        discount=(1.0 - done).astype(np.float32),
      )
      return timestep

    attrs["_to_dm"] = _to_dm
    return super().__new__(cls, name, parents, attrs)


def py_env(envspec: Type[EnvSpec],
           envpool: Type[EnvPool]) -> Tuple[Type[EnvSpec], Type[EnvPool]]:
  # remove the _ prefix added when registering cpp class via pybind
  spec_name = envspec.__name__[1:]
  pool_name = envpool.__name__[1:]
  return (
    EnvSpecMeta(spec_name, (envspec,), {}),  # type: ignore[return-value]
    EnvPoolMeta(pool_name, (envpool,), {}),
  )
