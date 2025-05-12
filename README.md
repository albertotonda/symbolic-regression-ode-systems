# Learning systems of Ordinary Differential Equations using standard Symbolic Regression

This repository contains the code and data for a series of works on learning ODE systems (system identification) using symbolic regression. Employing classical symbolic regression on dynamical systems requires applying a data transformation to the trajectory/time series data. Three different data transformations have been proposed in literature:
1. $Delta_x$
2. $F_x$
3. $\mathcal{C}_x$

## Testing data transformation approaches
The results reported in the publications below can be reproduced running scripts in Python and Julia. The Python scripts, comparing $Delta_x$ and $F_x$, can be called with:

```
cd utils
python check_odebench_all_transformations.py
```

## Publications
TONDA A., ZHANG H., CHEN Q., XUE B., ZHANG M., LUTTON E. 2025. When Data Transformations Mislead Symbolic Regression: Deceptive Search Spaces in System Identification, In: Workshop on Symbolic Regression, proceedings of the annual conference on Genetic and evolutionary computation (GECCO) 2025 companion, DOI: 10.1145/3712255.3734301  

TONDA A., ZHANG H., CHEN Q., XUE B., ZHANG M., LUTTON E. 2025. Comparing Data Transformation Techniques for System Identification With Standard Symbolic Regression, In: Proceedings of the annual conference on Genetic and evolutionary computation (GECCO) 2025 companion, DOI: 10.1145/3712255.3726673
