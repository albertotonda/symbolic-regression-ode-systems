# Learning systems of Ordinary Differential Equations using standard Symbolic Regression
Refactoring of the Symbolic Regression for systems of Ordinary Differential Equations project.

The original repository is here: https://github.com/albertotonda/eureqa-differential-equations

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

It would be nice for function #1 to finally return a pandas DataFrame. Same for function #2. 

## Benchmarks
Right now I have just a few benchmarks. But there are people who worked on a repository of benchmarks for ODEs. https://www.cse.chalmers.se/~dag/identification/Benchmarks/Problems.html

ODEBench is a modern version, with a lot of stuff pre-prepared by the people running ODEFormer.

## TODO
- Next step: build the systems to be optimized. 
- Also store the random seeds for each experiment.
- pySR has improved a lot in recent years, now it could work out of the box!
- perform a cross-validation, select only equations with recurring structure that appear in the Pareto fronts
