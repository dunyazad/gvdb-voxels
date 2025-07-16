#pragma once

#include <cuda_common.cuh>

#define EMPTY_KEY -1
#define VALID_KEY(k) ((k) != EMPTY_KEY)

template<typename Key, typename Value>
struct HashEntry
{
    Key key;
    Value value;
};

template<typename Key, typename Value>
struct HashMapInfo
{
    HashEntry<Key, Value>* entries = nullptr;
    size_t capacity = 1024 * 1024 * 1024;
    uint8_t maxProbe = 64;

    unsigned int* d_numberOccupiedKeys = nullptr;
    Key* d_occupiedKeys = nullptr;
};

template<typename Key, typename Value>
struct HashMap
{
    HashMapInfo<Key, Value> info;

    void Initialize(size_t capacity = 1024 * 1024 * 1024, uint8_t maxProbe = 64)
    {
        info.capacity = capacity;
        info.maxProbe = maxProbe;

        cudaMalloc(&info.entries, sizeof(HashEntry<Key, Value>) * info.capacity);
        cudaMemset(info.entries, 0xFF, sizeof(HashEntry<Key, Value>) * info.capacity);
    }

    void Terminate()
    {
        if (nullptr != info.entries)
        {
            cudaFree(info.entries);
        }
    }

    void Clear()
    {
        if (nullptr != info.entries)
        {
            cudaMemset(info.entries, 0xFF, sizeof(HashEntry<Key, Value>) * info.capacity);
        }
    }

    __device__ __host__
        inline size_t hash(Key key, size_t capacity) const
    {
        return static_cast<size_t>(key) % capacity;
    }

    __device__
        bool insert(HashMapInfo<Key, Value> info, Key key, Value value)
    {
        size_t idx = hash(key, info.capacity);

        for (int i = 0; i < info.maxProbe; ++i)
        {
            size_t slot = (idx + i) % info.capacity;
            Key* slot_key = &info.entries[slot].key;
            Key prev_key = atomicCAS(slot_key, EMPTY_KEY, key);

            if (prev_key == EMPTY_KEY || prev_key == key)
            {
                info.entries[slot].value = value;
                return true;
            }
        }
        return false;
    }

    __device__
        bool find(HashMapInfo<Key, Value> info, Key key, Value* outValue)
    {
        size_t idx = hash(key, info.capacity);

        for (int i = 0; i < info.maxProbe; ++i)
        {
            size_t slot = (idx + i) % info.capacity;
            Key k = info.entries[slot].key;

            if (k == key)
            {
                *outValue = info.entries[slot].value;
                return true;
            }
            if (k == EMPTY_KEY)
            {
                return false;
            }
        }
        return false;
    }
};
