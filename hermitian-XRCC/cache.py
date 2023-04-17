# stolen from adcc (and then modified)
#
# ^^^^ then we need to restore the original copyright and license notices!
# See; http://copresearch.pacific.edu/adutoi/group-resources.programming.openlicensing.html

from functools import wraps

def cached_member_function(function):
    """
    Decorates a member function being called with
    one or more arguments and stores the results
    in field `_function_cache` of the class instance.
    """
    fname = function.__name__

    @wraps(function)
    def wrapper(self, *args):
        try:
            fun_cache = self._function_cache[fname]
        except AttributeError:
            self._function_cache = {}
            fun_cache = self._function_cache[fname] = {}
        except KeyError:
            fun_cache = self._function_cache[fname] = {}

        try:
            return fun_cache[args]
        except KeyError:
            fun_cache[args] = function(self, *args)
            return fun_cache[args]
    return wrapper
