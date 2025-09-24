#include <SimpleBVH.cuh>

struct MortonKeyCompare {
    __host__ __device__
        bool operator()(const MortonKey& a, const MortonKey& b) const {
        if (a.code < b.code) return true;
        if (a.code > b.code) return false;
        if (a.index == b.index) printf("Error: duplicate morton key (code=%llu, index=%u)\n", a.code, a.index);
        return a.index < b.index;
    }
};
