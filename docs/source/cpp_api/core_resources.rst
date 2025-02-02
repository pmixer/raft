Resources
=========

.. role:: py(code)
   :language: c++
   :class: highlight

All resources which are specific to a computing environment like host or device are contained within, and managed by,
`raft::resources`. This design simplifies the APIs and eases user burden by making the APIs opaque by default but allowing customization based on user preference.


Vocabulary
----------

``#include <raft/core/resource/resource_types.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_types
     :project: RAFT
     :members:
     :content-only:


Device Resources
----------------

`raft::device_resources` is a convenience over using `raft::resources` directly. It provides accessor methods to retrieve resources such as the CUDA stream, stream pool, and handles to the various CUDA math libraries like cuBLAS and cuSOLVER. 

``#include <raft/core/device_resources.hpp>``

namespace *raft::core*

.. doxygenclass:: raft::device_resources
    :project: RAFT
    :members:


Resource Functions
------------------

Comms
~~~~~

``#include <raft/core/resource/comms.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_comms
     :project: RAFT
     :members:
     :content-only:

cuBLAS Handle
~~~~~~~~~~~~~

``#include <raft/core/resource/cublase_handle.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_cublas
     :project: RAFT
     :members:
     :content-only:

CUDA Stream
~~~~~~~~~~~

``#include <raft/core/resource/cuda_stream.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_cuda_stream
     :project: RAFT
     :members:
     :content-only:


CUDA Stream Pool
~~~~~~~~~~~~~~~~

``#include <raft/core/resource/cuda_stream_pool.hpp>``

namespace *raft::resource*

.. doxygengroup:: resource_stream_pool
    :project: RAFT
    :members:
    :content-only:

cuSolverDn Handle
~~~~~~~~~~~~~~~~~

``#include <raft/core/resource/cusolver_dn_handle.hpp>``
namespace *raft::resource*

 .. doxygengroup:: resource_cusolver_dn
     :project: RAFT
     :members:
     :content-only:

cuSolverSp Handle
~~~~~~~~~~~~~~~~~

``#include <raft/core/resource/cusolver_sp_handle.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_cusolver_sp
     :project: RAFT
     :members:
     :content-only:

cuSparse Handle
~~~~~~~~~~~~~~~

``#include <raft/core/resource/cusparse_handle.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_cusparse
     :project: RAFT
     :members:
     :content-only:

Device ID
~~~~~~~~~

``#include <raft/core/resource/device_id.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_device_id
     :project: RAFT
     :members:
     :content-only:


Device Memory Resource
~~~~~~~~~~~~~~~~~~~~~~

``#include <raft/core/resource/device_memory_resource.hpp>``

namespace *raft::resource*

 .. doxygengroup:: device_memory_resource
     :project: RAFT
     :members:
     :content-only:

Device Properties
~~~~~~~~~~~~~~~~~

``#include <raft/core/resource/device_properties.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_device_props
     :project: RAFT
     :members:
     :content-only:

Sub Communicators
~~~~~~~~~~~~~~~~~

``#include <raft/core/resource/sub_comms.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_sub_comms
     :project: RAFT
     :members:
     :content-only:

Thrust Exec Policy
~~~~~~~~~~~~~~~~~~

``#include <raft/core/resource/thrust_policy.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_thrust_policy
     :project: RAFT
     :members:
     :content-only:
