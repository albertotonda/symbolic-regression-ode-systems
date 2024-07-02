# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:13:05 2024

Local library with several utility functions, like opening and closing logging,
and calling functions with a time out.

@author: Alberto
"""

import functools
import threading
import sys
import time

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



if __name__ == "__main__" :
    
    # this code below is just a test of the timeout function wrapper
    def _fit_distribution():
        # some long operation which run for undetermined time
        for i in range(1):
            time.sleep(i)
            
        return 20
    
    try:
        return_value = timeout_func(_fit_distribution, timeout=2)
        print(return_value)
    except MyTimeoutError as ex:
        print(ex)