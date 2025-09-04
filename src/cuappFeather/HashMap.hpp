#pragma once

#include <cuda_runtime.h>
#include <cuda_common.cuh>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

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
	unsigned int* numberOfEntries = nullptr;
};

template<typename Key, typename Value>
struct HashMap
{
    HashMapInfo<Key, Value> info;

    void Initialize(size_t capacity = 1024 * 1024 * 1024, uint8_t maxProbe = 64, int value = 0xFF)
    {
        info.capacity = capacity;
        info.maxProbe = maxProbe;

        CUDA_CHECK(CUDA_MALLOC(&info.entries, sizeof(HashMapEntry<Key, Value>) * info.capacity));
        CUDA_CHECK(CUDA_MEMSET(info.entries, value, sizeof(HashMapEntry<Key, Value>) * info.capacity));

        CUDA_CHECK(CUDA_MALLOC(&info.numberOfEntries, sizeof(unsigned int)));
        CUDA_CHECK(CUDA_MEMSET(info.numberOfEntries, 0, sizeof(unsigned int)));
    }

    void Terminate()
    {
        if (info.entries != nullptr)
        {
            CUDA_CHECK(CUDA_FREE(info.entries));
            info.entries = nullptr;
        }

        if (info.numberOfEntries != nullptr)
        {
            CUDA_CHECK(CUDA_FREE(info.numberOfEntries));
            info.numberOfEntries = nullptr;
		}
    }

    void Clear(int value = 0xFF)
    {
        if (info.entries != nullptr)
        {
            CUDA_CHECK(CUDA_MEMSET(info.entries, value, sizeof(HashMapEntry<Key, Value>) * info.capacity));
        }

        if (info.numberOfEntries != nullptr)
        {
            CUDA_CHECK(CUDA_MEMSET(info.numberOfEntries, 0, sizeof(unsigned int)));
        }
    }

    __host__ __device__ static inline size_t HashMap_hash(Key key, size_t capacity)
    {
        return static_cast<size_t>(key) % capacity;
    }

    __device__ static bool insert(const HashMapInfo<Key, Value>& info, Key key, const Value& value)
    {
        size_t idx = HashMap_hash(key, info.capacity);

        for (int i = 0; i < info.maxProbe; ++i)
        {
            size_t slot = (idx + i) % info.capacity;
            Key* slot_key = &info.entries[slot].key;
            Key k = *slot_key;

            if (k == key)
            {
                // ÇÊ¿ä ½Ã atomicExch(&info.entries[slot].value, value);
                info.entries[slot].value = value;
                return true;

            }

            if (k == empty_key<Key>())
            {
                Key prev = atomicCAS(slot_key, empty_key<Key>(), key);
                if (prev == empty_key<Key>())
                {
					atomicAdd(info.numberOfEntries, 1);
                    info.entries[slot].value = value;
                    return true;
                }
                else if (prev == key)
                {
                    info.entries[slot].value = value;
                    return true;
                }
            }
        }
        return false;
    }

    __device__ static bool increase(const HashMapInfo<Key, Value>& info, Key key)
    {
        size_t idx = HashMap_hash(key, info.capacity);

        for (int i = 0; i < info.maxProbe; ++i)
        {
            size_t slot = (idx + i) % info.capacity;
            Key* slot_key = &info.entries[slot].key;
            Key k = *slot_key;

            if (k == key)
            {
                atomicAdd(&info.entries[slot].value, 1);
                return true;
            }

            if (k == empty_key<Key>())
            {
                Key prev = atomicCAS(slot_key, empty_key<Key>(), key);
                if (prev == empty_key<Key>())
                {
                    atomicAdd(info.numberOfEntries, 1);
                    info.entries[slot].value = 1;
                    return true;
                }
                else if (prev == key)
                {
                    atomicAdd(&info.entries[slot].value, 1);
                    return true;
                }
            }
        }
        return false;
    }

    __device__ static bool find(const HashMapInfo<Key, Value>& info, Key key, Value* outValue)
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
