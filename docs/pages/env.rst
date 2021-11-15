Add New Environment into EnvPool
================================

To add a new environment in C++ that will be parallelly run by EnvPool,
we provide a developer interface in `envpool/core/env.h
<https://github.com/sail-sg/envpool/blob/master/envpool/core/env.h>`_.

- For a quick and annotated example, please refer to
  `envpool/dummy/ <https://github.com/sail-sg/envpool/tree/master/envpool/dummy>`_.
- `envpool/atari
  <https://github.com/sail-sg/envpool/tree/master/envpool/atari>`_ serves as
  a more complex, real example.

In the following example, we will create an environment called ``Demo``


Create a Folder
---------------

The first thing is to create a ``demo`` folder under ``envpool/``:

.. code-block:: bash

    cd envpool
    mkdir -p demo
