////////////////////////////////////////////////////////////////////////////////
// Thymos Library Prelude (c) 2023-2026 by Zachary R. James (@thymos)
// Error and data type definitions only 
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define TH_MAX_DIMS 8

typedef enum {
    TH_OK = 0,
    TH_ERR_OOM,
    TH_ERR_INVALID,
    TH_ERR_SHAPE,
    TH_ERR_DTYPE,
    TH_ERR_UNSUPPORTED
} th_Error;

typedef enum {
    TH_DTYPE_INVALID = 0,
    TH_DTYPE_F32,
    TH_DTYPE_F64,
    TH_DTYPE_I32,
    TH_DTYPE_U8,
    TH_DTYPE_BOOL
} th_DType;
