"""
    Author          : Siddhi
    Created_date    : 2019/08/01
    Modified_date   :2019/08/06
    Description     : Package to solve linear problem
"""
import os

import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages

import random 
import numpy as np
import pandas as pd

rpy2.robjects.numpy2ri.activate()
# feasibility_dict = {0:'OPTIMAL', 1:'SUBOPIMAL', 2: 'INFEASIBLE', 3: 'UNBOUNDED'}

r_solver = '''
                function(matrix_coeff,sign, rhs, objective) {{
                sol = lpSolve::lp(direction = '{}', objective.in = objective, const.mat = matrix_coeff, 
                const.dir = sign, const.rhs = rhs, compute.sens = 2, scale = 1, transpose.constraints = TRUE)
                return (c(sol$status,sol$objval, sol$solution))
                            }}'''
                            

                            
class SmartSolver:
    """
    A class to solve linear programming problems. Includes solving by constraint relaxation 
    and adding new constraint (randomly) to solve unbounded problems.
    """
    def __init__(self, equation_dataframe, coefficient):
        """
            Initialize the class
            
            Parameters:
                equation_dataframe  (pandas.DataFrame)  : Equation dataframe. The rhs should be the last field, sign should come before rhs.
                coefficient         (list)              : coefficient of objective function
                direction           (str)               : min or max problem
        """
        self.equation_dataframe = equation_dataframe
        self.coefficient = coefficient
        self.matrix = None
        self.rhs = None
        self.rhs = None

    def create_new_equation_dataframe(self):
        """
        create new matrix from old matrix
        params      :sign_list: list of the sign of the constraints
    """
        try:
            matrix = pd.DataFrame(self.matrix)
            add_coeff = pd.DataFrame(np.identity(np.array(self.matrix).shape[0]))
            for ind, sign in enumerate(self.sign):
                if sign == '<=':
                    add_coeff.iloc[ind, :] = - add_coeff.iloc[ind, :]
            new_mat = pd.concat([matrix, add_coeff], axis = 1)
            new_mat['sign'] = self.sign
            new_mat['rhs'] = self.rhs
            new_mat['priority'] = self.priority
            return new_mat

        except Exception as e:
            raise e

    def get_input(self):
        """
            This functio is used to get the inputs for the optimizer, i.e coefficient matrix, 
            list of sign, list of rhs
            Note that fields priority, rhs and sign should be present in the equation dataframe 
            passed to the constructor.
            
            Returns:
                    (tuple) : (coefficient matrix, list of sign, list of rhs)
        """
        if  not set(['priority', 'rhs', 'sign']).issubset(self.equation_dataframe.columns):
            raise ValueError("Excpected 'priority', 'rhs', 'sign' in the equation dataframe.")
        self.priority = self.equation_dataframe['priority'].values.tolist()
        self.rhs = self.equation_dataframe['rhs'].values.tolist()
        self.sign = self.equation_dataframe['sign'].values.tolist()
        self.matrix = self.equation_dataframe.iloc[:,:-3].values.tolist()
        return self.matrix, self.sign, self.rhs
    
    def solve(self, direction = 'max'):
        if direction not in ['max', 'min']:
            raise ValueError("Expected min or max for parameter direction. Got {}".format(direction))
        """
            Use this function to solve the problem normally.
            
            Parameters:
                direction       (str)   : max or min
                
            Returns:
                                (tuple) : (infeasibility status, objective function, solution)
        """
        try:
            optimization_function = robjects.r(r_solver.format(direction))
            self.rhs = [float(x) for x in self.rhs]
            result_values = [item for item in optimization_function(np.array(self.matrix), self.sign, self.rhs, self.coefficient)]
            return result_values[0], result_values[1], result_values[2:]
        except Exception as e:
            raise e
        
    @staticmethod
    def get_random_array(matr):
        """
            Use this function to get a random constraint.
            
            Parameters:
                matr        (list, 2D)   : original equation matrix 
            
            Returns:
                            (numpy.array): list of new constraint
        """
        try:
            return np.random.uniform(-(max(max(matr))), (max(max(matr))), len(matr[0]))
        except Exception as e:
            raise e
        
    def get_new_input(self, add_sign):
        """
            Use this function to get the new equation matrix, rhs and sign. The primary
            use of this function is to provide random constraint.
            
            Parameters:
                add_sign    (str)   : sign to add to the newly formed constraint
                
            Returns:
                new input   (tuple) : (new equation matrix, new rhs list, new sign list)
        """
        
        if add_sign not in ['<=', '>=']:
            raise ValueError("Expected <= or >= for parameter add_sign. Got {}".format(add_sign))
        
        try:
            rand_arry = SmartSolver.get_random_array(self.matrix)
            self.matrix = self.matrix + [(rand_arry)]
            self.rhs = self.rhs+([random.randint(a = min(self.rhs), b = max(self.rhs))])
            self.sign = list(self.sign) + [add_sign]
            return np.array(self.matrix), self.rhs, self.sign
        except Exception as e:
            raise e
        
    def forced_unb(self, limit = 10, add_sign = '<=', direction = 'max'):
        """
            Use this function if you encountered unbounded function.
            This function generates synthetic constraints randomly and 
            iterates through each constraint until a feasible solution 
            is encountered or reaches certain limit.
            
            Parameters:
                limit       (int)   : number of iterations
                add_sign    (str)   : sign to add to the newly formed constraint
                direction   (str)   : min or max
            
            Returns:
                res         (tuple) : tuple of output of solve and the new dataframe
                
        """
        if direction not in ['max', 'min']:
            raise ValueError("Expected min or max for parameter direction. Got {}".format(direction))
        
        if add_sign not in ['<=', '>=']:
            raise ValueError("Expected min or max for parameter add_sign. Got {}".format(add_sign))
        
        try:
            status = 3
            count = 0
            while count < limit:
                self.get_input()
                new_matrix, new_rhs, new_sign = self.get_new_input(add_sign)
                res = self.solve(direction)
                eq_df = pd.DataFrame(new_matrix)
                eq_df['sign'] = new_sign
                eq_df['rhs'] = new_rhs
                eq_df.columns = self.equation_dataframe.columns[:-1]
                status = res[0]
                if status == 0: break
                count += 1
            return res, eq_df
        except Exception as e:
            raise e
        
    def forced_inf_rem_cons(self, direction, *args, **kwargs):
        """
            This function removes the constraints according to the spefied priority
            and starts solving the linear problem. If the solution is not found until
            the allowed iterations, this function just returns the solution and a df
            of removed constraints.
            
            Parameters:
                direction       : (str) either min or max
                args            : 
                kwargs          : keywords arguments such as limit (the number of iteration)
                
            Returns:
                result & removed
                constraints     : (tuple) (tuple from the solve method, df of removed constraints)
        """
        self.get_input()
        priority_list = list(set(self.priority))
        removed_constraints = []
        counter = 0
        result = None
        inf_eqn_df = self.equation_dataframe.copy(deep  = True)
        while 1:
            to_remove = max(priority_list)
            while counter < kwargs['limit']:
                priority_df = inf_eqn_df[inf_eqn_df['priority'] == to_remove]
                if len(priority_df)==0: 
                    break
                inf_solver = SmartSolver(inf_eqn_df, self.coefficient)
                inf_solver.get_input()
                inf_eqn_df = inf_eqn_df.drop(list(priority_df.index)[0], axis = 0).copy()
                removed_constraints.append(list(priority_df.index)[0])
                result = inf_solver.solve(direction)
                if result[0] == 0: 
                    return result , self.equation_dataframe.iloc[removed_constraints, :]
                counter +=1
            if counter == kwargs['limit']:break
            priority_list.remove(to_remove)
        return result, self.equation_dataframe.iloc[removed_constraints, :]
    
    @staticmethod    
    def create_new_coef(coef, num_static_constraints, multiplier=10000, reduce_coeff_by=1, reduce_coeff_for='none'):
        """
            create new objective function coefficients
            params      :coef : coefficient of the objective function
            params      :num_static_consntraints: number of static constraints
            params      :multiplier: the value that is to be multiplied to the added constraints
            params      :reduce_coeff_by: by how much factor the coefficient is to be reduced
            prams       :reduce_coeff_for: reduce the coefficeint for either min_thres or static_cons or none
            returns     : new objectice coefficient
        """
        try:
            if reduce_coeff_for == 'static_cons':
                return pd.concat(
                    [pd.DataFrame([coef]),
                    pd.DataFrame([(multiplier/reduce_coeff_by) *
                                np.ones(num_static_constraints)]),
                    pd.DataFrame([multiplier*np.ones(len(coef))])],
                    axis=1)
            elif reduce_coeff_for == 'min_thres':
                return pd.concat(
                    [pd.DataFrame([coef]),
                    pd.DataFrame([multiplier*np.ones(num_static_constraints)]),
                    pd.DataFrame([(multiplier/reduce_coeff_by)*np.ones(len(coef))])],
                    axis=1)

            elif reduce_coeff_for == 'none':
                return pd.concat(
                    [pd.DataFrame([coef]),
                    pd.DataFrame([multiplier*np.ones(num_static_constraints)]),
                    pd.DataFrame([(multiplier)*np.ones(len(coef))])],
                    axis=1)
            else:
                raise ValueError('''Incorrect value for parameter reduce_coeff_for.
                                Should be either min_thres or static_cons or none, got {}'''.format(reduce_coeff_for))

        except Exception as e:
            raise e



    def forced_inf_art_vars(self, direction, *args, **kwargs):
        """
            This function removes the constraints according to the spefied priority
            and starts solving the linear problem.
            
            Parameters:
                direction       : (str) either min or max
                args            : 
                kwargs          : (dict) keywords arguments such as multiplier, num_static_constraints 
                                        (parameters from create_new_coef)
            
            Returns:
               solution & new df: (tuple) (tuple returned by the solve method, new_dataframe with suggestions)
        """
        self.get_input()
        new_eq_df = self.create_new_equation_dataframe()
        if direction == 'max': kwargs['multiplier'] = -kwargs['multiplier']
        new_coefficients = SmartSolver.create_new_coef(self.coefficient, *args, **kwargs )
        inf_solver = SmartSolver(new_eq_df, new_coefficients.values)
        inf_solver.get_input()
        result = inf_solver.solve(direction)
        new_eq_df['suggested_corrections'] = result[2][len(self.coefficient):]
        new_eq_df['new_rhs'] = new_eq_df['rhs'] - new_eq_df['suggested_corrections']
        return result, new_eq_df
    
    def forced_inf(self, direction = 'max', method = 'rem_cons', *args, **kwargs):
        """
            Use this function if you encounter infeasible solution. The user has the
            facility to solve problem by removing constraints or by relaxing constraints
            (by introducing artificial variables.)
            
            Parameters:
                direction   :(str) either min or max
                method      :(str) either rem_cons or art_vars
                args        :(list)
                kwargs      :(dict)Pass the parameters in accordance to the method used
            
            Returns:
                solution according to the method used.
        """
        if method not in ['rem_cons', 'art_vars']:
            raise ValueError("Expected min or max for parameter method. Got {}".format(method))
        if direction not in ['min', 'max']:
            raise ValueError("Expected min or max for parameter direction. Got {}".format(direction))
        
        if method == 'rem_cons':
            return self.forced_inf_rem_cons(direction, **kwargs)
        else:
            return self.forced_inf_art_vars(direction, **kwargs)
        pass
        