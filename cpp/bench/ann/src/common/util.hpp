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

#include "ann_types.hpp"

#include "cuda_stub.hpp"

#include <sys/stat.h>
#include <sys/types.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sstream>
#include <string>
#include <vector>

#include <filesystem>

namespace raft::bench::ann {

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

inline auto cuda_info()
{
  int dev;
  cudaDeviceProp device_prop;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&device_prop, dev);
  std::vector<std::tuple<std::string, std::string>> props;
  props.emplace_back("gpu_name", std::string(device_prop.name));
  props.emplace_back("gpu_sm_count", std::to_string(device_prop.multiProcessorCount));
  props.emplace_back("gpu_sm_freq", std::to_string(device_prop.clockRate * 1e3));
  props.emplace_back("gpu_mem_freq", std::to_string(device_prop.memoryClockRate * 1e3));
  props.emplace_back("gpu_mem_bus_width", std::to_string(device_prop.memoryBusWidth));
  props.emplace_back("gpu_mem_global_size", std::to_string(device_prop.totalGlobalMem));
  props.emplace_back("gpu_mem_shared_size", std::to_string(device_prop.sharedMemPerMultiprocessor));
  return props;
}

inline std::vector<std::string> split(const std::string& s, char delimiter)
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream iss(s);
  while (getline(iss, token, delimiter)) {
    if (!token.empty()) { tokens.push_back(token); }
  }
  return tokens;
}

inline bool file_exists(const std::string& filename)
{
  struct stat statbuf;
  if (stat(filename.c_str(), &statbuf) != 0) { return false; }
  return S_ISREG(statbuf.st_mode);
}

inline bool dir_exists(const std::string& dir)
{
  struct stat statbuf;
  if (stat(dir.c_str(), &statbuf) != 0) { return false; }
  return S_ISDIR(statbuf.st_mode);
}

inline bool create_dir(const std::string& dir)
{
  const auto path = split(dir, '/');

  std::string cwd;
  if (!dir.empty() && dir[0] == '/') { cwd += '/'; }

  for (const auto& p : path) {
    cwd += p + "/";
    if (!dir_exists(cwd)) {
      int ret = mkdir(cwd.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
      if (ret != 0) { return false; }
    }
  }
  return true;
}

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
