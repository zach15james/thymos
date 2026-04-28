////////////////////////////////////////////////////////////////////////////////
// Thymos Library (c) 2023-2026 by Zachary R. James (@thymos)
//
// Thymos is a zero-dependency compositional compute runtime, first applied to ML.
//
// This file is the universal user-facing header (just include this, then build).
//
// Its design combines mechanical sympathy with mathematical fidelity: explicit
// C data structures, memory ownership, and thread-safe contexts on the machine
// side; typed compositional graphs, optimization routines, and compiler structure
// that stay close to the mathematics on the semantic side.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "prelude.h" // error, dtype, a few shared thingsc:w
#include "ctx.h"     // context = allocator (memory) + runtime (threads)
#include "tensor.h"  // tensor / shape / dtype runtime values
#include "act.h"     // graph IR / act layer
#include "opt.h"     // optimization defs, problems, & routines 
#include "df.h"      // dataframe + data processing
//#include "ml.h"
