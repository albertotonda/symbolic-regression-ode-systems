# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:13:05 2024

Local library with several utility functions, like opening and closing logging,
and calling functions with a time out.

@author: Alberto
"""

import functools
import multiprocessing
import threading
import sys
import time

from scipy import integrate

# unfortunately, all this MyTimeoutError/timeout_func does not work under
# Windows, causing mysterious "Windows fatal exception: access violation" errors
class MyTimeoutError(Exception):
    # exception for our timeouts
    pass

class InterruptableThread(threading.Thread):
    def __init__(self, func, args, kwargs, timeout, default) :
        threading.Thread.__init__(self)
        self.result = default
        self.exc_info = (None, None, None)
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.timeout = timeout

    def run(self) :
        try :
            self.result = self.func(*(self.args or ()), **(self.kwargs or {}))
        except Exception as err :
            self.exc_info = sys.exc_info()

    def suicide(self) :
        raise MyTimeoutError(
            "{0} timeout (taking more than {1} seconds)".format(self.func.__name__, self.timeout)
        )

def timeout_func(func, args=None, kwargs=None, timeout=30, default=None) :
    """This function will spawn a thread and run the given function
    using the args, kwargs and return the given default value if the
    timeout is exceeded.
    http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
    
    Timeout is specified in seconds (or fractions thereof, it can be a float)
    """
    it = InterruptableThread(func, args, kwargs, timeout, default)
    it.start()
    it.join(timeout)
    
    # this part below might not be 100% correct, but it kinda works;
    # however, on Windows I am sometimes getting weird errors on memory access
    #if it.exc_info[0] is not None :
    #    a, b, c = it.exc_info
    #    raise Exception(a, b, c)  # communicate that to caller

    if it.is_alive() :
        it.suicide()
        raise RuntimeError
    else:
        return it.result

# end of the part that does not work

def timeout(seconds_before_timeout):
    """
    This is another way of approaching the problem, using a decorator.
    Usage: 
        - either the classic @timeout before defining the function
        - or func = timeout(timeout=16)(MyModule.MyFunc), and then try/except
    """
    def deco(func) :
        @functools.wraps(func)
        def wrapper(*args, **kwargs) :
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, seconds_before_timeout))]
            def newFunc() :
                try :
                    res[0] = func(*args, **kwargs)
                except Exception as e :
                    res[0] = e
            t = threading.Thread(target=newFunc)
            t.daemon = True
            try :
                t.start()
                t.join(seconds_before_timeout)
            except Exception as e :
                print('error starting thread')
                raise e
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco


def process_function(queue, *args, **kwargs) :
    print("Process starting")
    result = integrate.odeint(*args, **kwargs)
    print("Process finished")
    queue.put(result)
    
    return

def example_function(queue, argument, kwargument="ciao") :
    
    result = str(argument) + " " + str(kwargument)
    queue.put(result)
    
    return

def odeint_process_wrapper(conn, dX_dt, initial_conditions, t, odeint_args=(), full_output=False) :
    print(dX_dt)
    print(initial_conditions)
    print(t)
    print(odeint_args)
    print(full_output)
    result = integrate.odeint(dX_dt, initial_conditions, t, args=odeint_args, full_output=full_output)
    Y, info_dict = result
    print(Y)
    
    #queue.put(Y)
    conn.send(result)
    
    return

def run_process_with_timeout(func, timeout, func_args=None, func_kwargs=None) :
    
    with multiprocessing.Pool(processes=1) as pool :
        result = pool.apply_async(func, func_args, func_kwargs)
        try :
            return result.get(timeout=timeout)
        except multiprocessing.TimeoutError :
            pool.terminate()
            raise TimeoutError("Timeout exceeded.")
    
    return

if __name__ == "__main__" :
    
    import pandas as pd
    
    from sympy import parse_expr
    from optimize_ode_systems import dX_dt
    
    # this code below is just a test of the timeout function wrapper
    df_ode = pd.read_csv("2024-07-03-13-15-55-lotka-volterra/lotka-volterra.csv")
    variables_list = ["x", "y"]
    initial_conditions = [df_ode[v].values[0] for v in variables_list]
    t = df_ode["t"].values
    
    equations = {'x': "-0.017114721*y", 'y': "-0.49482120130712*x"}
    candidate_equations = {variables_list[i] : parse_expr(equations[variables_list[i]])
                           for i in range(0, len(variables_list))}
    
    args = (dX_dt, initial_conditions, t)
    kwargs = {'args' : (candidate_equations, variables_list, ),
              'full_output' : True}
    
    # all right, another attempt: fork a process
    result = None
    try:
        result = run_process_with_timeout(integrate.odeint, 10, args, kwargs)
        print(result)
    except TimeoutError as e:
        print(e)
    except Exception as e:
        print("Exception:", str(e))
    
    # all this part did not work
    # #queue = multiprocessing.Queue() # queues have issues with large data, let's use pipes!
    # parent_conn, child_conn = multiprocessing.Pipe()
    
    # print("Calling function...")
    # process = multiprocessing.Process(target=odeint_process_wrapper, 
    #                                   args=(child_conn,) + args, 
    #                                   kwargs=kwargs)
    
    # process.start()
    # process.join(20)
    
    # result = "process failed"
    
    # if process.is_alive() :
    #     process.terminate()
    #     process.join()
    #     print("Timeout exceeded.")
    # else :
    #     if parent_conn.poll() :
    #         result = parent_conn.recv()
    
    # print(result)