import time


def timeit(func):
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()

        print 'Function: %s.%r - time elapsed: %2.4f sec' % (args[0].__class__.__name__, func.__name__, te - ts)
        return result

    return timed
