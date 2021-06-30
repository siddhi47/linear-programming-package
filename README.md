<h1>SmartSolver</h1>
The SmartSolver class to solve linear programming problems. The underlying library of SmartSolver is an R based linear programming solver called 'lpSolve'. The R program is called by the ExtensoSolver to solve a linear program.

The maximization and minimization problems are pretty straight forward. However, the user has additional facility to use tackle with infeasible as well as unbounded solution.

<h2>Infeasible solution</h2>
To tackle with infeasible solution, the user can now chose from two alternatives:
<ul>
    <li> Introduce artificial variables to solve the problem (relaxing constraits).</li>
    <li> Remove the constraints one by one to identify the constraint that is causing the infeasibility.</li>
</ul>

<h2>Unbounded solutions</h2>
The unbounded solutions exists in a linear programming problem probably because of:
<li>Wrong constraint formulation.
<li> The problem is the opposite type of problem (i.e. if you are trying a maximization problem, then it is probably a minimization problem and vice versa)
<li> There are not enough constraints.

The unbounded solution cannot just be handled by introducing additional variable, because there is no gurantee that there will be a bounded region after introducing such variables. Instead what we can do is, add another constraint hoping the added constraint will introduce a boundary. This, is the basic idea behind treating unbounded solution in linear programming problem.

<h1>Using ExtensoSolver</h1>
<h2>Setting up environment</h2>

<li> R (preferably 3.5.2) should be installed in the system.
<li> Python (preferably 3.6.7, python 2 is strongly not recommended) should be installed in the system. 
<li> Set up a virtual environment, install the requirements in the virtual environment.
<>

<h2>Installing requirements</h2>
The requirements.txt file is inside the lib folder. Find it and install the libraries. It is highly recommended to create and use a virtual environment.

```
            virtualenv venv
            source venv\bin\activate
    (venv)  pip install -r requirements.txt
```
**NOTE : The libraries (rpy2) only work with linux environment. The Libraries may not install with a windows machine.** 

<h2>Import the class and instantiate</h2>

```python
from smart_solver import SmartSolver

solver = SmartSolver(equation_df, coef)
```

<h2>Getting the appropriate inputs</h2>

```python
# the inputs are calculated internally and saved as attribute of the object as well.
matrix, sign, rhs = solver.get_inputs()
```

<h2>Using solve method</h2>

```python
solution = solver.solve('max')
```

<h2>Solving unbounded solutions</h2>

```python
self.solver.get_input()
# maximization problem. The number of new random constraints are limited by limit parameter. The add_sign parameter is the direction for the newly formed constraint
solver.forced_unb(limit = 10, add_sign = '<=', direction = 'max')
```

<h2>Solving infeasible solutions</h2>

```python
# There are two methods to solve infeasible problems. One is to remove constraint and another one is to relax the constraints. The  static constraints are all the constraints except the thresholds.

solver.forced_inf('max', 'rem_cons', limit = 20)
solver.forced_inf('max', 'art_vars',  num_static_constraints = 5, multiplier = 1000)
```

