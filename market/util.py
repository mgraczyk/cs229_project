import inspect
import lzma
import pickle
import os

def memoized_zero(f):
    computed = False
    value = None
    def memozero():
        nonlocal computed, value
        if not computed:
            value = f()
            computed = True
        return value
    return memozero

def memoized(f):
   nargs = len(inspect.getargspec(f)[0])
   if nargs == 0:
       memofunc = memoized_zero(f)
   elif nargs == 1:
      class memodict(dict):
         """ Memoization decorator for a function taking a single argument """
         def __missing__(self, key):
            self[key] = ret = f(key)
            return ret
      memofunc = memodict().__getitem__
   else:
      class memodict(dict):
         """ Memoization decorator for a function taking multiple arguments """
         def __missing__(self, key):
            self[key] = ret = f(*key)
            return ret

      getter = memodict().__getitem__
      def get_packed(*key):
          return getter(key)

      memofunc = get_packed

   return memofunc

def dict_factory(cursor, row):
    """ Dictionary factory for database access.
    """
    return { col[0]: row[idx]
            for idx, col in enumerate(cursor.description) }

def pickle_save(obj, name):
    os.makedirs("./pickle", exist_ok=True)
    with lzma.open("./pickle/{}".format(name), "wb") as f:
        pickle.dump(obj, f)

def pickle_load(name):
    with lzma.open("./pickle/{}".format(name), "rb") as f:
        return pickle.load(f)
