#pragma once

// Vector containers
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Algorithms
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/pair.h>
#include <thrust/partition.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cuda_runtime.h>

// ---------------- Entry ----------------
template <typename Key, typename Value>
struct HashMapEntry {
    Key key;
    Value value;
    int occupied = 0;
    int tombstone = 0;
};

// ---------------- Device Hash ----------------
template <typename Key>
__device__ __forceinline__ size_t HashDevice(Key key, size_t capacity) {
    return static_cast<size_t>(key) * 2654435761 % capacity;
}
template <>
__device__ __forceinline__ size_t HashDevice<float>(float key, size_t capacity) {
    return static_cast<size_t>(__float_as_int(key)) * 2654435761 % capacity;
}
template <>
__device__ __forceinline__ size_t HashDevice<double>(double key, size_t capacity) {
    return static_cast<size_t>(__double_as_longlong(key)) * 2654435761 % capacity;
}

// ---------------- GPU Kernels ----------------
template <typename Key, typename Value>
__global__ void InsertKernel(HashMapEntry<Key, Value>* entries, size_t capacity,
    const Key* keys, const Value* values, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    Key key = keys[idx];
    Value value = values[idx];
    size_t hash = HashDevice(key, capacity);

    for (size_t i = 0; i < capacity; i++) {
        size_t pos = (hash + i) % capacity;

        if (entries[pos].occupied && entries[pos].key == key) {
            entries[pos].value = value;
            break;
        }

        if ((entries[pos].occupied == 0) && atomicCAS(&entries[pos].occupied, 0, 1) == 0) {
            entries[pos].key = key;
            entries[pos].value = value;
            entries[pos].tombstone = 0;
            break;
        }
    }
}

// Find Kernel
template <typename Key, typename Value>
__global__ void FindKernel(HashMapEntry<Key, Value>* entries, size_t capacity,
    const Key* keys, Value* results, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    Key key = keys[idx];
    Value result{};
    bool found = false;
    size_t hash = HashDevice(key, capacity);

    for (size_t i = 0; i < capacity; i++) {
        size_t pos = (hash + i) % capacity;
        if (entries[pos].occupied && entries[pos].key == key) {
            result = entries[pos].value;
            found = true;
            break;
        }
    }
    results[idx] = found ? result : Value{};
}

// Remove Kernel
template <typename Key, typename Value>
__global__ void RemoveKernel(HashMapEntry<Key, Value>* entries, size_t capacity,
    const Key* keys, size_t N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    Key key = keys[idx];
    size_t hash = HashDevice(key, capacity);

    for (size_t i = 0; i < capacity; i++) {
        size_t pos = (hash + i) % capacity;
        if (entries[pos].occupied && entries[pos].key == key) {
            entries[pos].tombstone = 1;
            entries[pos].occupied = 0;
            break;
        }
    }
}

// ---------------- HashMap Class ----------------
template <typename Key, typename Value>
class HashMap {
public:
    HashMap(size_t capacity = 1 << 20) // 충분히 큰 capacity
        : capacity_(capacity)
    {
        entries.resize(capacity_);
        thrust::fill(thrust::device, entries.begin(), entries.end(), HashMapEntry<Key, Value>{});
    }

    void insert(const std::vector<Key>& h_keys, const std::vector<Value>& h_values)
    {
        assert(h_keys.size() == h_values.size());
        size_t N = h_keys.size();

        thrust::device_vector<Key> d_keys = h_keys;
        thrust::device_vector<Value> d_values = h_values;

        int threads = 256;
        int blocks = (N + threads - 1) / threads;

        InsertKernel << <blocks, threads >> > (thrust::raw_pointer_cast(entries.data()), capacity_,
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_values.data()), N);
        cudaDeviceSynchronize();
        size_ += N;
    }

    void find(const std::vector<Key>& h_keys, std::vector<Value>& h_results)
    {
        size_t N = h_keys.size();
        h_results.resize(N);

        thrust::device_vector<Key> d_keys = h_keys;
        thrust::device_vector<Value> d_results(N);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        FindKernel << <blocks, threads >> > (thrust::raw_pointer_cast(entries.data()), capacity_,
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_results.data()), N);
        cudaDeviceSynchronize();

        thrust::copy(d_results.begin(), d_results.end(), h_results.begin());
    }

    void remove(const std::vector<Key>& h_keys)
    {
        size_t N = h_keys.size();
        thrust::device_vector<Key> d_keys = h_keys;

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        RemoveKernel << <blocks, threads >> > (thrust::raw_pointer_cast(entries.data()), capacity_,
            thrust::raw_pointer_cast(d_keys.data()), N);
        cudaDeviceSynchronize();
    }

    size_t capacity() const { return capacity_; }

private:
    thrust::device_vector<HashMapEntry<Key, Value>> entries;
    size_t capacity_;
    size_t size_ = 0;
};
