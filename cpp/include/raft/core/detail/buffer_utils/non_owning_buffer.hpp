/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <raft/core/mdspan.hpp>
#include <raft/core/memory_type.hpp>

namespace raft {
namespace detail {
template <typename ElementType,
          memory_type M,
          typename Extents,
          typename LayoutPolicy = layout_c_contiguous>
struct non_owning_buffer {

  non_owning_buffer() : data_{nullptr} {}

  non_owning_buffer(ElementType* ptr, Extents extents) : data_{ptr}, extents_{extents} {
  }

  auto* data_handle() const { return data_; }

  auto* view() {
    bool device_accessible = is_device_accessible(M);
    bool host_accessible = is_host_accessible(M);
    return make_mdspan<ElementType, LayoutPolicy, host_accessible, device_accessible>(data_, extents_);
  }
 private:
  ElementType* data_;
  Extents extents_;
};

}  // namespace detail
}  // namespace raft