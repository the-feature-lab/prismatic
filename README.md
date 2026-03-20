# flab

Library for Feature Lab empirics.

## Structure

```
flab/
├── core/               # General-purpose ML research utilities
│   ├── devices.py          # Device management and tensor/array conversion
│   ├── empirics.py         # Plotting, experiment measurements (ExptTrace), and file I/O (FileManager)
│   ├── imagedata.py        # Image dataset loading, downloading, and preprocessing
│   └── krr.py              # Kernel ridge regression and eigenlearning theory
│
└── prismatic/          # Tools for polynomial/Hermite feature analysis
    ├── data.py             # Synthetic data generation (power-law, polynomial targets)
    ├── feature_decomp.py   # Monomial representation and HEA eigenvalue computation
    ├── models.py           # Neural network architectures (MLP, CNN, ExpanderMLP)
    └── utils.py            # Lower-level array/tensor utilities and seeding
```
