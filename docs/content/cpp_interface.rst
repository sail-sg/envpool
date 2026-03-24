C++ API Reference
=================

This reference is generated from the curated ``envpool/core`` headers used by
the :doc:`new_env` integration guide. It is meant to make the C++ extension
surface available on Read the Docs alongside the narrative guide.


Core Data Structures
--------------------

.. doxygenclass:: Array
   :project: envpool_cpp_api

.. doxygenclass:: TArray
   :project: envpool_cpp_api

.. doxygenclass:: ShapeSpec
   :project: envpool_cpp_api

.. doxygenclass:: Spec
   :project: envpool_cpp_api

The compile-time dictionary helpers used by these types still live in
``envpool/core/dict.h`` and are referenced throughout :doc:`new_env`.


Environment Authoring
---------------------

.. doxygenvariable:: common_config
   :project: envpool_cpp_api

.. doxygenvariable:: common_action_spec
   :project: envpool_cpp_api

.. doxygenvariable:: common_state_spec
   :project: envpool_cpp_api

.. doxygenclass:: EnvSpec
   :project: envpool_cpp_api

.. doxygenclass:: Env
   :project: envpool_cpp_api


Pool Implementations
--------------------

.. doxygenclass:: EnvPool
   :project: envpool_cpp_api

.. doxygenclass:: AsyncEnvPool
   :project: envpool_cpp_api


Python Binding Helpers
----------------------

.. doxygenclass:: PyEnvSpec
   :project: envpool_cpp_api

.. doxygenclass:: PyEnvPool
   :project: envpool_cpp_api
