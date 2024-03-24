import platform, os, ctypes

LIB: ctypes.CDLL = None

def _find_lib(extension: str) -> ctypes.CDLL:
    shared_lib_path: str = os.path.join(os.path.dirname(__file__), f"sharedlib/libSNNPy.{extension}")
    if not os.path.isfile(shared_lib_path):
        raise FileNotFoundError(f"Shared library not found at {shared_lib_path}")
    return ctypes.CDLL(shared_lib_path)
    
def _verify_lib(lib: ctypes.CDLL) -> None:
    if lib.runTests() != 0:
        raise RuntimeError("Library tests failed")

# Defines initialization logic
def _initialize():
    os_type: str = platform.system()

    if os_type == "Darwin":
        LIB = _find_lib("dylib")
        _verify_lib(LIB)
        
    elif os_type == "Windows":
        LIB = ctypes.CDLL = _find_lib("dll")
        _verify_lib(LIB)

    else :
        raise OSError(f"Unsupported OS: {os_type}")
    
# Call the initialization function when the module is imported
_initialize()
