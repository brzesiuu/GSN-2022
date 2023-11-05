import functools
from time import time


def time_measure(func=None, *, title=None):
    if func is None:
        return functools.partial(time_measure, title=title)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(f"Time of {title}: {time() - start}")
        return result

    return wrapper
