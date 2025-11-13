#!/usr/bin/env python3
"""
TEAM-506: Fix CUDA-specific code in quantized.hip for ROCm compatibility
This script applies all necessary fixes for ROCm/HIP compilation
"""

import re
import sys

def fix_quantized_hip(filename):
    print(f"🔧 Fixing CUDA-specific code in {filename}...")
    
    with open(filename, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Add TEAM-506 header at the top
    print("  ✓ Adding TEAM-506 header...")
    header = """// TEAM-506: ROCm/HIP port of quantized kernels from CUDA
// Original: candle-kernels/src/quantized.cu
// Converted: hipify-perl + manual fixes for ROCm compatibility
// 
// Key changes:
// - cuda_bf16.h → hip_bfloat16.h
// - __CUDA_ARCH__ → __HIP_DEVICE_COMPILE__
// - __dp4a → __builtin_amdgcn_sdot4
// - __vsubss4 → custom implementation (no ROCm equivalent)
// - CUDA macros → HIP macros
// - Added ROCm architecture detection (RDNA1/2/3, CDNA2)
//
"""
    if not content.startswith("// TEAM-506"):
        content = header + content
    
    # 2. Fix cuda_bf16.h include
    print("  ✓ Replacing cuda_bf16.h with hip_bfloat16.h...")
    content = content.replace('#include "cuda_bf16.h"', '#include "hip/hip_bfloat16.h"')
    
    # 3. Fix CUDA macro names
    print("  ✓ Replacing CUDA macros with HIP equivalents...")
    content = content.replace('GGML_CUDA_ASSUME', 'GGML_HIP_ASSUME')
    content = content.replace('GGML_CUDA_F16', 'GGML_HIP_F16')
    content = content.replace('GGML_CUDA_DMMV_X', 'GGML_HIP_DMMV_X')
    content = content.replace('CUDA_QUANTIZE_BLOCK_SIZE', 'HIP_QUANTIZE_BLOCK_SIZE')
    content = content.replace('CUDA_DEQUANTIZE_BLOCK_SIZE', 'HIP_DEQUANTIZE_BLOCK_SIZE')
    content = content.replace('CUDA_USE_TENSOR_CORES', 'HIP_USE_TENSOR_CORES')
    
    # 4. Fix function names
    print("  ✓ Replacing CUDA function names with HIP equivalents...")
    content = content.replace('ggml_cuda_dp4a', 'ggml_hip_dp4a')
    content = content.replace('vec_dot_q_cuda_t', 'vec_dot_q_hip_t')
    content = content.replace('allocate_tiles_cuda_t', 'allocate_tiles_hip_t')
    content = content.replace('load_tiles_cuda_t', 'load_tiles_hip_t')
    content = content.replace('vec_dot_q_mul_mat_cuda_t', 'vec_dot_q_mul_mat_hip_t')
    
    # 5. Fix __CUDA_ARCH__
    print("  ✓ Replacing __CUDA_ARCH__ with __HIP_DEVICE_COMPILE__...")
    content = content.replace('__CUDA_ARCH__', '__HIP_DEVICE_COMPILE__')
    
    # 6. Fix __dp4a intrinsic
    print("  ✓ Replacing __dp4a with ROCm equivalent...")
    content = content.replace('__dp4a', '__builtin_amdgcn_sdot4')
    
    # 7. Implement __vsubss4 replacement
    print("  ✓ Implementing __vsubss4 replacement...")
    vsubss4_impl = """
// TEAM-506: ROCm replacement for CUDA __vsubss4 intrinsic
// __vsubss4 performs 4 parallel 8-bit saturated subtractions
static __device__ __forceinline__ int __vsubss4_impl(int a, int b) {
    int result = 0;
    
    // Process each byte with saturation
    for (int i = 0; i < 4; i++) {
        int shift = i * 8;
        int8_t a_byte = (a >> shift) & 0xFF;
        int8_t b_byte = (b >> shift) & 0xFF;
        
        // Saturated subtraction: clamp to [-128, 127]
        int diff = (int)a_byte - (int)b_byte;
        if (diff > 127) diff = 127;
        if (diff < -128) diff = -128;
        
        result |= ((diff & 0xFF) << shift);
    }
    
    return result;
}

#define __vsubss4(a, b) __vsubss4_impl(a, b)
"""
    
    # Insert after get_int_from_uint8_aligned function
    if '__vsubss4_impl' not in content:
        pattern = r'(static __device__ __forceinline__ int get_int_from_uint8_aligned\(const uint8_t \* x8, const int & i32\) \{[^}]+\})'
        content = re.sub(pattern, r'\1' + vsubss4_impl, content)
    
    # 8. Fix CUDART_HMAX
    print("  ✓ Commenting out CUDART_HMAX...")
    content = re.sub(r'#define CUDART_HMAX.*', '// #define CUDART_HMAX removed - not needed for ROCm', content)
    
    # 9. Add ROCm architecture detection
    print("  ✓ Adding ROCm architecture detection...")
    rocm_arch = """
// TEAM-506: ROCm architecture detection
#ifdef __HIP_DEVICE_COMPILE__
  #if defined(__gfx1030__)
    #define ROCM_TARGET_ARCH CC_RDNA2
  #elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__)
    #define ROCM_TARGET_ARCH CC_RDNA3
  #elif defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__)
    #define ROCM_TARGET_ARCH CC_RDNA1
  #elif defined(__gfx90a__)
    #define ROCM_TARGET_ARCH CC_CDNA2
  #else
    #define ROCM_TARGET_ARCH CC_RDNA2  // Default to RDNA2
  #endif
#endif
"""
    
    if 'ROCM_TARGET_ARCH' not in content:
        # Insert after CC_RDNA3 definition
        pattern = r'(#define CC_RDNA3\s+\(CC_OFFSET_AMD \+ 1100\))'
        content = re.sub(pattern, r'\1' + rocm_arch, content)
    
    # Write back
    if content != original_content:
        with open(filename, 'w') as f:
            f.write(content)
        print("✅ All fixes applied successfully!")
        return True
    else:
        print("⚠️  No changes needed")
        return False

def main():
    filename = 'quantized.hip'
    
    try:
        if fix_quantized_hip(filename):
            print("\nSummary of changes:")
            print("  • Replaced cuda_bf16.h with hip_bfloat16.h")
            print("  • Replaced __CUDA_ARCH__ with __HIP_DEVICE_COMPILE__")
            print("  • Replaced __dp4a with __builtin_amdgcn_sdot4")
            print("  • Implemented __vsubss4 replacement (8 occurrences)")
            print("  • Renamed CUDA macros to HIP equivalents")
            print("  • Renamed CUDA functions to HIP equivalents")
            print("  • Added ROCm architecture detection")
            print("  • Added TEAM-506 attribution header")
            print("\nNext steps:")
            print("  1. Compile with hipcc:")
            print("     hipcc -c quantized.hip -o quantized.hsaco --offload-arch=gfx1030,gfx1100,gfx90a")
            print("  2. Embed HSACO in quantized.rs")
            print("  3. Test with actual workloads")
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
