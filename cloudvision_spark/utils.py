import io
import logging
import numpy as np
import resource
import os

from pympler import muppy, summary


log_dir = "/tmp/spark-executors/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(filename=log_dir + "all.log",
                    level=logging.DEBUG,
                    format='%(asctime)s %(message)s')



def serialize_numpy_array(numpy_array):
    output = io.BytesIO()
    np.savez_compressed(output, x=numpy_array)
    return output.getvalue()


def deserialize_numpy_array(savez_data):
    arrays = io.BytesIO(savez_data)
    data = np.load(arrays)
    return data["x"]


def log(msg):
    logging.info(msg)

def log_memory_usage():
    memory_used_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    pid = os.getpid()
    log('Memory usage of pid(%s): %s (kb)' % (pid, memory_used_kb))
#    all_objects = muppy.get_objects()
#    log("Total amount of objects: %s" % len(all_objects))
#    sum1 = summary.summarize(all_objects)
#    for line in summary.format_(sum1):
#        log(line)
