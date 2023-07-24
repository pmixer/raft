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

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include <filesystem>

namespace raft::bench::ann {

enum class MemoryType {
  Host,
  HostMmap,
  Device,
};

template <typename T>
struct buf {
  MemoryType memory_type;
  std::size_t size;
  T* data;
  buf(MemoryType memory_type, std::size_t size)
    : memory_type(memory_type), size(size), data(nullptr)
  {
    switch (memory_type) {
      case MemoryType::Device: {
        cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T));
        cudaMemset(data, 0, size * sizeof(T));
      } break;
      default: {
        data = reinterpret_cast<T*>(malloc(size * sizeof(T)));
        std::memset(data, 0, size * sizeof(T));
      }
    }
  }
  ~buf() noexcept
  {
    if (data == nullptr) { return; }
    switch (memory_type) {
      case MemoryType::Device: {
        cudaFree(data);
      } break;
      default: {
        free(data);
      }
    }
  }

  [[nodiscard]] auto move(MemoryType target_memory_type) -> buf<T>
  {
    buf<T> r{target_memory_type, size};
    if ((memory_type == MemoryType::Device && target_memory_type != MemoryType::Device) ||
        (memory_type != MemoryType::Device && target_memory_type == MemoryType::Device)) {
      cudaMemcpy(r.data, data, size * sizeof(T), cudaMemcpyDefault);
    } else {
      std::swap(data, r.data);
    }
    return r;
  }
};

struct cuda_timer {
 private:
  cudaStream_t stream_;
  cudaEvent_t start_;
  cudaEvent_t stop_;
  double total_time_{0};

 public:
  struct cuda_lap {
   private:
    cudaStream_t stream_;
    cudaEvent_t start_;
    cudaEvent_t stop_;
    double& total_time_;

   public:
    cuda_lap(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop, double& total_time)
      : start_(start), stop_(stop), stream_(stream), total_time_(total_time)
    {
      cudaStreamSynchronize(stream_);
      cudaEventRecord(start_, stream_);
    }
    cuda_lap() = delete;

    ~cuda_lap() noexcept
    {
      cudaEventRecord(stop_, stream_);
      cudaEventSynchronize(stop_);
      float milliseconds = 0.0f;
      cudaEventElapsedTime(&milliseconds, start_, stop_);
      total_time_ += milliseconds / 1000.0;
    }
  };

  cuda_timer()
  {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    cudaEventCreate(&stop_);
    cudaEventCreate(&start_);
  }

  ~cuda_timer() noexcept
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
    cudaStreamDestroy(stream_);
  }

  [[nodiscard]] auto stream() const -> cudaStream_t { return stream_; }

  [[nodiscard]] auto total_time() const -> double { return total_time_; }

  [[nodiscard]] auto lap() -> cuda_timer::cuda_lap
  {
    return cuda_lap{stream_, start_, stop_, total_time_};
  }
};

std::vector<std::string> split(const std::string& s, char delimiter);

bool file_exists(const std::string& filename);
bool dir_exists(const std::string& dir);
bool create_dir(const std::string& dir);

inline void make_sure_parent_dir_exists(const std::string& file_path)
{
  const auto pos = file_path.rfind('/');
  if (pos != std::string::npos) {
    auto dir = file_path.substr(0, pos);
    if (!dir_exists(dir)) { create_dir(dir); }
  }
}

inline auto combine_path(const std::string& dir, const std::string& path)
{
  std::filesystem::path p_dir(dir);
  std::filesystem::path p_suf(path);
  return (p_dir / p_suf).string();
}

template <typename... Ts>
void log_(const char* level, const Ts&... vs)
{
  char buf[20];
  std::time_t now = std::time(nullptr);
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
  printf("%s [%s] ", buf, level);
  if constexpr (sizeof...(Ts) == 1) {
    printf("%s", vs...);
  } else {
    printf(vs...);
  }
  printf("\n");
  fflush(stdout);
}

template <typename... Ts>
void log_info(Ts&&... vs)
{
  log_("info", std::forward<Ts>(vs)...);
}

template <typename... Ts>
void log_warn(Ts&&... vs)
{
  log_("warn", std::forward<Ts>(vs)...);
}

template <typename... Ts>
void log_error(Ts&&... vs)
{
  log_("error", std::forward<Ts>(vs)...);
}

}  // namespace raft::bench::ann
