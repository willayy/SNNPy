import platform, os, ctypes



def _find_lib(extension: str) -> ctypes.CDLL:
    shared_lib_path: str = os.path.join(os.path.dirname(__file__), f"sharedlib//libSNNPy.{extension}")
    if not os.path.isfile(shared_lib_path):
        raise FileNotFoundError(f"Shared library not found at {shared_lib_path}")
    c_lib = ctypes.CDLL(shared_lib_path)
    return c_lib
    
def _verify_lib(lib: ctypes.CDLL) -> None:
    if lib.runTests() != 0:
        raise RuntimeError("Library tests failed")

# Defines initialization logic
def _initialize():
    os_type: str = platform.system()
    global c_lib 
    if os_type == "Darwin":
        c_lib = _find_lib("dylib")
        _verify_lib(c_lib)
        
    elif os_type == "Windows":
        c_lib = ctypes.CDLL = _find_lib("dll")
        _verify_lib(c_lib)

    else :
        raise OSError(f"Unsupported OS: {os_type}")
    
# Call the initialization function when the module is imported
_initialize()
