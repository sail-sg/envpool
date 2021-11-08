.. EnvPool documentation master file, created by
   sphinx-quickstart on Mon Oct 25 21:01:07 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EnvPool!
===================

EnvPool is a C++ based batched environment pool with pybind11 and threadpool.
It has high performance (~1M raw FPS in DGX on Atari games) and compatible APIs
(supports both gym and dm_env, both sync and async).

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Installation
------------

.. code-block:: bash

   $ pip install envpool

After installation, open your python console and type

::

    import envpool
    print(envpool.__version__)

If no error occurs, you have successfully installed EnvPool.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
