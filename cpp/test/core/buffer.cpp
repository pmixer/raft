/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <raft/core/buffer.hpp>
#include <raft/core/execution_stream.hpp>
#include <raft/core/memory_type.hpp>
#include <raft/core/exceptions.hpp>

namespace raft {

// TEST(Buffer, default_buffer)
// {
//   auto buf = buffer<int>();
//   EXPECT_EQ(buf.mem_type(), memory_type::host);
//   EXPECT_EQ(buf.size(), 0);
//   EXPECT_EQ(buf.device_index(), 0);
// }

// TEST(Buffer, device_buffer)
// {
//   auto data = std::vector<int>{1, 2, 3};
//   auto test_buffers = std::vector<buffer<int>>{};
//   test_buffers.emplace_back(data.size(), memory_type::device, 0, execution_stream{});
//   test_buffers.emplace_back(data.size(), memory_type::device, 0);
//   test_buffers.emplace_back(data.size(), memory_type::device);

//   for (auto& buf : test_buffers) {
//     ASSERT_EQ(buf.mem_type(), memory_type::device);
//     ASSERT_EQ(buf.size(), data.size());
// #ifndef RAFT_DISABLE_CUDA
//     ASSERT_NE(buf.data(), nullptr);

//     auto data_out = std::vector<int>(data.size());
//     cudaMemcpy(static_cast<void*>(buf.data()),
//                static_cast<void*>(data.data()),
//                sizeof(int) * data.size(),
//                cudaMemcpyHostToDevice);
//     cudaMemcpy(static_cast<void*>(data_out.data()),
//                static_cast<void*>(buf.data()),
//                sizeof(int) * data.size(),
//                cudaMemcpyDeviceToHost);
//     EXPECT_THAT(data_out, testing::ElementsAreArray(data));
// #endif
//   }
// }

// TEST(Buffer, non_owning_device_buffer)
// {
//   auto data = std::vector<int>{1, 2, 3};
//   auto* ptr_d = static_cast<int*>(nullptr);
// #ifndef RAFT_DISABLE_CUDA
//   cudaMalloc(reinterpret_cast<void**>(&ptr_d), sizeof(int) * data.size());
//   cudaMemcpy(static_cast<void*>(ptr_d),
//              static_cast<void*>(data.data()),
//              sizeof(int) * data.size(),
//              cudaMemcpyHostToDevice);
// #endif
//   auto test_buffers = std::vector<buffer<int>>{};
//   test_buffers.emplace_back(ptr_d, data.size(), memory_type::device, 0);
//   test_buffers.emplace_back(ptr_d, data.size(), memory_type::device);
// #ifndef RAFT_DISABLE_CUDA

//   for (auto& buf : test_buffers) {
//     ASSERT_EQ(buf.mem_type(), memory_type::device);
//     ASSERT_EQ(buf.size(), data.size());
//     ASSERT_EQ(buf.data(), ptr_d);

//     auto data_out = std::vector<int>(data.size());
//     cudaMemcpy(static_cast<void*>(data_out.data()),
//                static_cast<void*>(buf.data()),
//                sizeof(int) * data.size(),
//                cudaMemcpyDeviceToHost);
//     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//   }
//   cudaFree(reinterpret_cast<void*>(ptr_d));
// #endif
// }

// TEST(Buffer, host_buffer)
// {
//   auto data   = std::vector<int>{1, 2, 3};
//   auto test_buffers = std::vector<buffer<int>>{};
//   test_buffers.emplace_back(data.size(), memory_type::host, 0, execution_stream{});
//   test_buffers.emplace_back(data.size(), memory_type::host, 0);
//   test_buffers.emplace_back(data.size(), memory_type::host);
//   test_buffers.emplace_back(data.size());

//   for (auto& buf : test_buffers) {
//     ASSERT_EQ(buf.mem_type(), memory_type::host);
//     ASSERT_EQ(buf.size(), data.size());
//     ASSERT_NE(buf.data(), nullptr);

//     std::memcpy(
//       static_cast<void*>(buf.data()), static_cast<void*>(data.data()), data.size() * sizeof(int));

//     auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
//     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//   }
// }

// TEST(Buffer, host_buffer_from_iters)
// {
//   auto data   = std::vector<int>{1, 2, 3};
//   auto test_buffers = std::vector<buffer<int>>{};
//   test_buffers.emplace_back(std::begin(data), std::end(data));

//   for (auto& buf : test_buffers) {
//     ASSERT_EQ(buf.mem_type(), memory_type::host);
//     ASSERT_EQ(buf.size(), data.size());
//     ASSERT_NE(buf.data(), nullptr);

//     std::memcpy(
//       static_cast<void*>(buf.data()), static_cast<void*>(data.data()), data.size() * sizeof(int));

//     auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
//     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//   }
// }

// TEST(Buffer, device_buffer_from_iters)
// {
//   auto data = std::vector<int>{1, 2, 3};
//   auto test_buffers = std::vector<buffer<int>>{};
//   test_buffers.emplace_back(std::begin(data), std::end(data), memory_type::device);
//   test_buffers.emplace_back(std::begin(data), std::end(data), memory_type::device, 0);
//   test_buffers.emplace_back(std::begin(data), std::end(data), memory_type::device, 0, execution_stream{});

//   for (auto& buf : test_buffers) {
//     ASSERT_EQ(buf.mem_type(), memory_type::device);
//     ASSERT_EQ(buf.size(), data.size());
// #ifndef RAFT_DISABLE_CUDA
//     ASSERT_NE(buf.data(), nullptr);

//     auto data_out = std::vector<int>(data.size());
//     cudaMemcpy(static_cast<void*>(buf.data()),
//                static_cast<void*>(data.data()),
//                sizeof(int) * data.size(),
//                cudaMemcpyHostToDevice);
//     cudaMemcpy(static_cast<void*>(data_out.data()),
//                static_cast<void*>(buf.data()),
//                sizeof(int) * data.size(),
//                cudaMemcpyDeviceToHost);
//     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
// #endif
//   }
// }

TEST(Buffer, non_owning_host_buffer)
{
  auto data   = std::vector<int>{1, 2, 3};
  std::vector<buffer<int>> test_buffers;
  test_buffers.emplace_back(data.data(), data.size(), memory_type::host, 0);
  // ASSERT_EQ(test_buffers.back().mem_type(), memory_type::host);
  //   ASSERT_EQ(test_buffers.back().size(), data.size());
  //   ASSERT_EQ(test_buffers.back().data(), data.data());
  test_buffers.emplace_back(data.data(), data.size(), memory_type::host);
  // ASSERT_EQ(test_buffers.back().mem_type(), memory_type::host);
  //   ASSERT_EQ(test_buffers.back().size(), data.size());
  //   ASSERT_EQ(test_buffers.back().data(), data.data());
  test_buffers.emplace_back(data.data(), data.size());
  // ASSERT_EQ(test_buffers.back().mem_type(), memory_type::host);
  //   ASSERT_EQ(test_buffers.back().size(), data.size());
  //   ASSERT_EQ(test_buffers.back().data(), data.data());

  // for (auto& buf : test_buffers) 
  for (int i = 0; i < 3; i++) {
    RAFT_LOG_INFO("memory_type %d\n", test_buffers[i].mem_type());
    ASSERT_EQ(test_buffers[i].mem_type(), memory_type::host);
    ASSERT_EQ(test_buffers[i].size(), data.size());
    ASSERT_EQ(test_buffers[i].data(), data.data());

    // auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
    // EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
  }
}

// TEST(Buffer, copy_buffer)
// {
//   auto data        = std::vector<int>{1, 2, 3};
//   auto orig_buffer = buffer<int>(data.data(), data.size(), memory_type::host);

//   auto test_buffers = std::vector<buffer<int>>{};
//   test_buffers.emplace_back(orig_buffer);
//   test_buffers.emplace_back(orig_buffer, memory_type::host);
//   test_buffers.emplace_back(orig_buffer, memory_type::host, 0);
//   test_buffers.emplace_back(orig_buffer, memory_type::host, 0, execution_stream{});

//   for (auto& buf : test_buffers) {
//     ASSERT_EQ(buf.mem_type(), memory_type::host);
//     ASSERT_EQ(buf.size(), data.size());
//     ASSERT_NE(buf.data(), orig_buffer.data());

//     auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
//     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));

// #ifndef RAFT_DISABLE_CUDA
//     auto test_dev_buffers = std::vector<buffer<int>>{};
//     test_dev_buffers.emplace_back(orig_buffer, memory_type::device);
//     test_dev_buffers.emplace_back(orig_buffer, memory_type::device, 0);
//     test_dev_buffers.emplace_back(orig_buffer, memory_type::device, 0, execution_stream{});
//     for (auto& dev_buf : test_dev_buffers) {
//       data_out = std::vector<int>(data.size());
//       RAFT_CUDA_TRY(cudaMemcpy(static_cast<void*>(data_out.data()), static_cast<void*>(dev_buf.data()), dev_buf.size() * sizeof(int), cudaMemcpyDefault));
//       EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));

//       auto test_dev_copies = std::vector<buffer<int>>{};
//       test_dev_copies.emplace_back(dev_buf, memory_type::device);
//       test_dev_copies.emplace_back(dev_buf, memory_type::device, 0);
//       test_dev_copies.emplace_back(dev_buf, memory_type::device, 0, execution_stream{});
//       for (auto& copy_buf : test_dev_copies) {
//         data_out = std::vector<int>(data.size());
//         RAFT_CUDA_TRY(cudaMemcpy(static_cast<void*>(data_out.data()), static_cast<void*>(copy_buf.data()), copy_buf.size() * sizeof(int), cudaMemcpyDefault));
//         EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//       }

//       auto test_host_buffers = std::vector<buffer<int>>{};
//       test_host_buffers.emplace_back(dev_buf, memory_type::host);
//       test_host_buffers.emplace_back(dev_buf, memory_type::host, 0);
//       test_host_buffers.emplace_back(dev_buf, memory_type::host, 0, execution_stream{});
//       for (auto& host_buf : test_host_buffers) {
//         data_out = std::vector<int>(host_buf.data(), host_buf.data() + host_buf.size());
//         EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//       }
//     }
// #endif
//   }
// }

// TEST(Buffer, move_buffer)
// {
//   auto data   = std::vector<int>{1, 2, 3};
//   auto test_buffers = std::vector<buffer<int>>{};
//   test_buffers.emplace_back(buffer<int>(data.data(), data.size(), memory_type::host));
//   test_buffers.emplace_back(buffer<int>(data.data(), data.size(), memory_type::host), memory_type::host);
//   test_buffers.emplace_back(buffer<int>(data.data(), data.size(), memory_type::host), memory_type::host, 0);
//   test_buffers.emplace_back(buffer<int>(data.data(), data.size(), memory_type::host), memory_type::host, 0, execution_stream{});

//   for (auto& buf : test_buffers) {
//     ASSERT_EQ(buf.mem_type(), memory_type::host);
//     ASSERT_EQ(buf.size(), data.size());
//     ASSERT_EQ(buf.data(), data.data());

//     auto data_out = std::vector<int>(buf.data(), buf.data() + buf.size());
//     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//   }
// #ifndef RAFT_DISABLE_CUDA
//   test_buffers = std::vector<buffer<int>>{};
//   test_buffers.emplace_back(buffer<int>(data.data(), data.size(), memory_type::host), memory_type::device);
//   test_buffers.emplace_back(buffer<int>(data.data(), data.size(), memory_type::host), memory_type::device, 0);
//   test_buffers.emplace_back(buffer<int>(data.data(), data.size(), memory_type::host), memory_type::device, 0, execution_stream{});
//   for (auto& buf : test_buffers) {
//     ASSERT_EQ(buf.mem_type(), memory_type::device);
//     ASSERT_EQ(buf.size(), data.size());
//     ASSERT_NE(buf.data(), data.data());

//     auto data_out = std::vector<int>(buf.size());
//     RAFT_CUDA_TRY(cudaMemcpy(static_cast<void*>(data_out.data()), static_cast<void*>(buf.data()), buf.size() * sizeof(int), cudaMemcpyDefault));
//     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//   }
// #endif
// }

// TEST(Buffer, move_assignment_buffer)
// {
//   auto data = std::vector<int>{1, 2, 3};

// #ifndef RAFT_DISABLE_CUDA
//   auto buf = buffer<int>{data.data(), data.size() - 1, memory_type::device};
// #else
//   auto buf = buffer<int>{data.data(), data.size() - 1, memory_type::host};
// #endif
//   buf      = buffer<int>{data.size(), memory_type::host};

//   ASSERT_EQ(buf.mem_type(), memory_type::host);
//   ASSERT_EQ(buf.size(), data.size());
// }

// TEST(Buffer, partial_buffer_copy)
// {
//   auto data1 = std::vector<int>{1, 2, 3, 4, 5};
//   auto data2 = std::vector<int>{0, 0, 0, 0, 0};
//   auto expected = std::vector<int>{0, 3, 4, 5, 0};
// #ifndef RAFT_DISABLE_CUDA
//   auto buf1 = buffer<int>{buffer<int>{data1.data(), data1.size(), memory_type::host}, memory_type::device};
// #else
//   auto buf1 = buffer<int>{data1.data(), data1.size(), memory_type::host};
// #endif
//   auto buf2 = buffer<int>{data2.data(), data2.size(), memory_type::host};
//   copy<true>(buf2, buf1, 1, 2, 3, execution_stream{});
//   copy<false>(buf2, buf1, 1, 2, 3, execution_stream{});
//   EXPECT_THROW(copy<true>(buf2, buf1, 1, 2, 4, execution_stream{}), out_of_bounds);
// }

// TEST(Buffer, buffer_copy_overloads)
// {
//   auto data        = std::vector<int>{1, 2, 3};
//   auto expected = data;
//   auto orig_host_buffer = buffer<int>(data.data(), data.size(), memory_type::host);
//   auto orig_dev_buffer = buffer<int>(orig_host_buffer, memory_type::device);
//   auto copy_dev_buffer = buffer<int>(data.size(), memory_type::device);
  
//   // copying host to host
//   auto data_out = std::vector<int>(data.size());
//   auto copy_host_buffer = buffer<int>(data_out.data(), data.size(), memory_type::host);
//   copy<true>(copy_host_buffer, orig_host_buffer);
//   EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));

//   // copying host to host with stream
//   data_out = std::vector<int>(data.size());
//   copy_host_buffer = buffer<int>(data_out.data(), data.size(), memory_type::host);
//   copy<true>(copy_host_buffer, orig_host_buffer, execution_stream{});
//   EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));

//   // copying host to host with offset
//   data_out = std::vector<int>(data.size() + 1);
//   copy_host_buffer = buffer<int>(data_out.data(), data.size(), memory_type::host);
//   copy<true>(copy_host_buffer, orig_host_buffer, 2, 1, 1, execution_stream{});
//   expected = std::vector<int>{0, 0, 2, 0};
//   EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));

// #ifndef RAFT_DISABLE_CUDA
//   // copy device to host
//   data_out = std::vector<int>(data.size());
//   copy_host_buffer = buffer<int>(data_out.data(), data.size(), memory_type::host);
//   copy<true>(copy_host_buffer, orig_dev_buffer);
//   expected = data;
//   EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));

//   // copy device to host with stream
//   data_out = std::vector<int>(data.size());
//   copy_host_buffer = buffer<int>(data_out.data(), data.size(), memory_type::host);
//   copy<true>(copy_host_buffer, orig_dev_buffer, execution_stream{});
//   expected = data;
//   EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));
  
//   // copy device to host with offset
//   data_out = std::vector<int>(data.size() + 1);
//   copy_host_buffer = buffer<int>(data_out.data(), data.size(), memory_type::host);
//   copy<true>(copy_host_buffer, orig_dev_buffer, 2, 1, 1, execution_stream{});
//   expected = std::vector<int>{0, 0, 2, 0};
//   EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));
// #endif
// }

}