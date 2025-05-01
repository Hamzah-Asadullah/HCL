#include <cstdint>
#include <cstdlib>

#ifndef ALIGNED_MALLOC_CPP
#define ALIGNED_MALLOC_CPP

namespace HCL
{
    template <typename T>
    T* simple_aligned_malloc(std::size_t block_size, std::size_t size)
    {
        if ((block_size < alignof(std::max_align_t)) || ((block_size & (block_size - 1)) != 0))
            return nullptr;
    
        void* raw = std::malloc(size + block_size + sizeof(void*));
        if (!raw) return nullptr;
    
        std::uintptr_t raw_addr = reinterpret_cast<std::uintptr_t>(raw) + sizeof(void*);
        std::uintptr_t aligned_addr = (raw_addr + block_size - 1) & ~(block_size - 1);
        void** aligned_ptr = reinterpret_cast<void**>(aligned_addr);
        aligned_ptr[-1] = raw;
    
        return reinterpret_cast<T*>(aligned_ptr);
    }
    
    template <typename T>
    void simple_aligned_free(T* aligned)
    {
        if (aligned)
            std::free(reinterpret_cast<void**>(aligned)[-1]);
    }
};

#endif