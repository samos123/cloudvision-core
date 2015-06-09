import io
import numpy as np


def serialize_numpy_array(numpy_array):
    output = io.BytesIO()
    np.savez_compressed(output, x=numpy_array)
    return output.getvalue()


def deserialize_numpy_array(savez_data):
    arrays = io.BytesIO(savez_data)
    data = np.load(arrays)
    return data["x"]
