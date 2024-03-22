import platform, os, ctypes

# Defines initialization logic
def _initialize():
    os_type: str = platform.system()

    if os_type == "Darwin":
        _find_lib("dylib")
        
    elif os_type == "Windows":
        _find_lib("dll")
        
    else :
        raise OSError(f"Unsupported OS: {os_type}")
    
def _find_lib(extension: str) -> None:
    shared_lib_path: str = os.path.join(os.path.dirname(__file__), f"sharedlib/libSNNPy.{extension}")
    if not os.path.isfile(shared_lib_path):
        raise FileNotFoundError(f"Shared library not found at {shared_lib_path}")
    if ctypes.CDLL(shared_lib_path).runTests() != 0:
        raise RuntimeError("Library tests failed")

    
# Call the initialization function when the module is imported
_initialize()
