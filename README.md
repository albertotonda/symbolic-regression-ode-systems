# Learning systems of Ordinary Differential Equations using standard Symbolic Regression
Refactoring of the Symbolic Regression for systems of Ordinary Differential Equations project.

## NOTES
I need several scripts and functions:
1. One that reads a .txt (or better, json) file with a system of differential equations, plus initial conditions, and generates data (see src/createDataset/createDataset.py).
2. One that transforms the dataset by using the 'trick' in the paper, the approximate Eulerian form (see src/createDeltat/createDeltat.py).
3. Run separate instances of pySR on each equation, obtaining several Pareto fronts of candidate solutions.
4. Pair equations found by pySR, converting them back to their original dx/dt format (ODE systems).
5. Identify parameters inside the ODE systems, and optimize them using CMA-ES against the original data.
6. Return the best ODE system.

Ideally, a lot of experiments can be run, changing:
- noise applied to the data
- regular/irregular sampling

## TODO
- Next step: build the systems to be optimized. 
- Also store the random seeds for each experiment.
- Creating a conda environment with everything included seems to be super painful. Either I installed pySR, or all the other package. At the moment, environment (srode) seems to work. Fingers crossed.
