#    (C) Copyright 2018, 2019 Anthony D. Dutoi
# 
#    This file is part of Qode.
# 
#    Qode is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
# 
#    Qode is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License
#    along with Qode.  If not, see <http://www.gnu.org/licenses/>.
#

from functools import wraps

# stolen from adcc (and then modified)
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
