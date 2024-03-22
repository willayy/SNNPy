import ctypes

def _get_lib() -> ctypes.CDLL:
    shared_lib_path: str = os.path.join(os.path.dirname(__file__), "sharedlib/libSNNPy.dylib")
    return ctypes.CDLL(shared_lib_path)
    
    