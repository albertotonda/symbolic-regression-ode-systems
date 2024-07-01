# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:13:05 2024

Local library with several utility functions, like opening and closing logging,
and calling functions with a time out.

@author: Alberto
"""

# test_threading.py
import threading
import sys
import time

class MyTimeoutError(Exception):
    # exception for our timeouts
    pass

def timeout_func(func, args=None, kwargs=None, timeout=30, default=None):
    """This function will spawn a thread and run the given function
    using the args, kwargs and return the given default value if the
    timeout is exceeded.
    http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
    
    Timeout is specified in seconds (or fractions thereof, it can be a float)
    """

    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = default
            self.exc_info = (None, None, None)

        def run(self):
            try:
                self.result = func(*(args or ()), **(kwargs or {}))
            except Exception as err:
                self.exc_info = sys.exc_info()

        def suicide(self):
            raise MyTimeoutError(
                "{0} timeout (taking more than {1} sec)".format(func.__name__, timeout)
            )

    it = InterruptableThread()
    it.start()
    it.join(timeout)

    if it.exc_info[0] is not None:
        a, b, c = it.exc_info
        raise Exception(a, b, c)  # communicate that to caller

    if it.is_alive():
        it.suicide()
        raise RuntimeError
    else:
        return it.result


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