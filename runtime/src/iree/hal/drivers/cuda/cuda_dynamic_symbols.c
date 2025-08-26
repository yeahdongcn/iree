// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/cuda_dynamic_symbols.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/hal/drivers/cuda/cuda_status_util.h"

static const char* iree_hal_cuda_dylib_names[] = {
#if !defined(IREE_MUSA_IS_CUDA)
#if defined(IREE_PLATFORM_WINDOWS)
    "nvcuda.dll",
#else
    "libcuda.so",
#endif  // IREE_PLATFORM_WINDOWS
#else
    "libmusa.so",
#endif    // IREE_MUSA_IS_CUDA
};

// CUDA API version for cuGetProcAddress.
// 1000 * major + 10 * minor
#define IREE_CUDA_DRIVER_API_VERSION 11030

// Load CUDA entry points.
static iree_status_t iree_hal_cuda_dynamic_symbols_resolve_all(
    iree_hal_cuda_dynamic_symbols_t* syms) {
  // Since cuGetProcAddress is in the symbol table, it will be loaded again
  // through cuGetProcAddress. cuGetProcAddress_v2 is added in CUDA 12.0 and has
  // a new function signature. If IREE_CUDA_DRIVER_API_VERSION is increased to
  // >=12.0, then make sure we are using the correct signature.
#if !defined(IREE_MUSA_IS_CUDA)
  IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(
      syms->dylib, "cuGetProcAddress", (void**)&syms->cuGetProcAddress));
#define IREE_CU_PFN_DECL(cuda_symbol_name, ...)                         \
  {                                                                     \
    static const char* name = #cuda_symbol_name;                        \
    IREE_CUDA_RETURN_IF_ERROR(                                          \
        syms,                                                           \
        cuGetProcAddress(name, (void**)&syms->cuda_symbol_name,         \
                         IREE_CUDA_DRIVER_API_VERSION,                  \
                         CU_GET_PROC_ADDRESS_DEFAULT),                  \
        "when resolving " #cuda_symbol_name " using cuGetProcAddress"); \
  }
#else
IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(
      syms->dylib, "muInit", (void**)&syms->cuInit));
  syms->cuInit(0);
#define IREE_CU_PFN_DECL(cuda_symbol_name, ...)                                 \
  {                                                                             \
    static char name[256];                                                      \
    snprintf(name, sizeof(name), "mu%s", #cuda_symbol_name + 2);                \
    struct {                                                                    \
      const char* orig;                                                         \
      const char* v2;                                                           \
    } v2_map[] = {                                                              \
      {"muCtxCreate",               "muCtxCreate_v2"},                          \
      {"muCtxDestroy",              "muCtxDestroy_v2"},                         \
      {"muDevicePrimaryCtxRelease", "muDevicePrimaryCtxRelease_v2"},            \
      {"muCtxPushCurrent",          "muCtxPushCurrent_v2"},                     \
      {"muCtxPopCurrent",           "muCtxPopCurrent_v2"},                      \
      {"muEventDestroy",            "muEventDestroy_v2"},                       \
      {"muGraphInstantiate",        "muGraphInstantiate_v2"},                   \
      {"muMemAlloc",                "muMemAlloc_v2"},                           \
      {"muMemFree",                 "muMemFree_v2"},                            \
      {"muMemHostRegister",         "muMemHostRegister_v2"},                    \
      {"muMemHostGetDevicePointer", "muMemHostGetDevicePointer_v2"},            \
      {"muMemcpyHtoDAsync",         "muMemcpyHtoDAsync_v2"},                    \
      {"muStreamDestroy",           "muStreamDestroy_v2"},                      \
    };                                                                          \
    for (size_t i = 0; i < sizeof(v2_map) / sizeof(v2_map[0]); ++i) {           \
      if (strcmp(name, v2_map[i].orig) == 0) {                                  \
        strcpy(name, v2_map[i].v2);                                             \
        break;                                                                  \
      }                                                                         \
    }                                                                           \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(                    \
        syms->dylib, name, (void**)&syms->cuda_symbol_name));                   \
  }
#endif    // IREE_MUSA_IS_CUDA
#include "iree/hal/drivers/cuda/cuda_dynamic_symbol_table.h"  // IWYU pragma: keep
#undef IREE_CU_PFN_DECL
  return iree_ok_status();
}

iree_status_t iree_hal_cuda_dynamic_symbols_initialize(
    iree_allocator_t host_allocator,
    iree_hal_cuda_dynamic_symbols_t* out_syms) {
  IREE_ASSERT_ARGUMENT(out_syms);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_syms, 0, sizeof(*out_syms));
  iree_status_t status = iree_dynamic_library_load_from_files(
      IREE_ARRAYSIZE(iree_hal_cuda_dylib_names), iree_hal_cuda_dylib_names,
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &out_syms->dylib);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "CUDA driver library 'libcuda.so'/'nvcuda.dll' not available; please "
        "ensure installed and in dynamic library search path");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda_dynamic_symbols_resolve_all(out_syms);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_cuda_dynamic_symbols_deinitialize(out_syms);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_cuda_dynamic_symbols_deinitialize(
    iree_hal_cuda_dynamic_symbols_t* syms) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_dynamic_library_release(syms->dylib);
  memset(syms, 0, sizeof(*syms));

  IREE_TRACE_ZONE_END(z0);
}
