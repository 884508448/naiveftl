from time import time
from utils.ftl_log import LOGGER
from functools import wraps

def timer(func):
    @wraps(func)
    def decorated(*args,**kwargs):
        start = time()
        r=func(*args,**kwargs)
        time_cost = time()-start
        LOGGER.info(f"time cost: {time_cost}s")
        return r
    return decorated