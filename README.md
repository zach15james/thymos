### Thymos

The 3 levels:
(i) category-theoretic graphical design
(ii) compiler
(iii) kernel implementation

yet the use case will be:
(1) WHAT (user-facing, delcarative)
ie opt.h, models/, quant/, tools/, ...
(2) How (compiler, passes, fusion)
ie th_Graph: TypePool, MorphPool, EdgePool
then th_comple to passes + th_Plan
(3) WHERE/WHEN (kernels & backends)
th_run: walks tape, dispatches via registry
th_KernelRegistry: KernelID x Capatility --> apply_fn {TPU/LPU/NPU, CPU(naive, AVX2, BLAS), CUDA/Metal/ROCm}

th_Context := { struct th_Memory + th_Execution }


currently no working version:
```easy command to immediately get a valuable run```
