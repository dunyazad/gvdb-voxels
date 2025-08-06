#pragma once

#include <cuda_common.cuh>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

// Type-safe EMPTY_KEY definition
template<typename Key>
__host__ __device__ inline Key empty_key();

template<>
__host__ __device__ inline int empty_key<int>()
{
    return -1;
}

template<>
__host__ __device__ inline uint64_t empty_key<uint64_t>()
{
    return 0xFFFFFFFFFFFFFFFFULL;
}

// Hash entry structure
template<typename Key, typename Value>
struct HashMapEntry
{
    Key key;
    Value value;
};

template<typename Key, typename Value>
struct HashMapInfo
{
    HashMapEntry<Key, Value>* entries = nullptr;
    size_t capacity = 1024 * 1024 * 1024;
    uint8_t maxProbe = 64;

    unsigned int* d_numberOccupiedKeys = nullptr;
    Key* d_occupiedKeys = nullptr;
};

// CUDA error check macro (for robustness)
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) \
        { \
            printf("CUDA error %s (%d): %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
            assert(false); \
        } \
    } while (0)

template<typename Key, typename Value>
struct HashMap
{
    HashMapInfo<Key, Value> info;

    void Initialize(size_t capacity = 1024 * 1024 * 1024, uint8_t maxProbe = 64)
    {
        info.capacity = capacity;
        info.maxProbe = maxProbe;

        CUDA_CHECK(cudaMalloc(&info.entries, sizeof(HashMapEntry<Key, Value>) * info.capacity));
        CUDA_CHECK(cudaMemset(info.entries, 0xFF, sizeof(HashMapEntry<Key, Value>) * info.capacity));
    }

    void Terminate()
    {
        if (info.entries != nullptr)
        {
            CUDA_CHECK(cudaFree(info.entries));
            info.entries = nullptr;
        }
    }

    void Clear(int value = 0xFF)
    {
        if (info.entries != nullptr)
        {
            CUDA_CHECK(cudaMemset(info.entries, value, sizeof(HashMapEntry<Key, Value>) * info.capacity));
        }
    }

    __host__ __device__ inline size_t HashMap_hash(Key key, size_t capacity) const
    {
        return static_cast<size_t>(key) % capacity;
    }

    __device__ bool insert(const HashMapInfo<Key, Value>& info, Key key, const Value& value)
    {
        size_t idx = HashMap_hash(key, info.capacity);

        for (int i = 0; i < info.maxProbe; ++i)
        {
            size_t slot = (idx + i) % info.capacity;
            Key* slot_key = &info.entries[slot].key;
            Key prev_key = atomicCAS(slot_key, empty_key<Key>(), key);

            if (prev_key == empty_key<Key>() || prev_key == key)
            {
                info.entries[slot].value = value;
                return true;
            }
        }
        return false;
    }

    __device__ bool find(const HashMapInfo<Key, Value>& info, Key key, Value* outValue)
    {
        size_t idx = HashMap_hash(key, info.capacity);

        for (int i = 0; i < info.maxProbe; ++i)
        {
            size_t slot = (idx + i) % info.capacity;
            Key k = info.entries[slot].key;

            if (k == key)
            {
                *outValue = info.entries[slot].value;
                return true;
            }
            if (k == empty_key<Key>())
            {
                return false;
            }
        }
        return false;
    }
};

#undef CUDA_CHECK
