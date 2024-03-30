import platform, os, ctypes as ct
from snnpy.neuralnetwork import NeuralNetwork

def _find_lib(extension: str) -> ct.CDLL:
    shared_lib_path: str = os.path.join(os.path.dirname(__file__), f"sharedlib//libSNN.{extension}")
    if not os.path.isfile(shared_lib_path):
        raise FileNotFoundError(f"Shared library not found at {shared_lib_path}")
    c_lib = ct.CDLL(shared_lib_path)
    return c_lib
    
def _verify_lib(lib: ct.CDLL) -> None:
    if lib.runTests() != 0:
        raise RuntimeError("Library tests failed")
    
def _initialize_lib(c_lib: ct.CDLL) -> None:
    c_lib.initNeuralNetwork.argtypes = [ct.POINTER(NeuralNetwork), ct.c_int, ct.c_int, ct.c_int, ct.c_int]
    c_lib.initNeuralNetwork.restype = None
    c_lib.trainNeuralNetworkOnBatch.argtypes = [ct.POINTER(NeuralNetwork), ct.POINTER(ct.POINTER(ct.c_double)), 
                                                ct.POINTER(ct.POINTER(ct.c_double)), ct.c_int, ct.c_int, 
                                                ct.c_double, ct.c_double, ct.c_double, ct.c_int]
    c_lib.trainNeuralNetworkOnBatch.restype = None
    c_lib.inputDataToNeuralNetwork.argtypes = [ct.POINTER(NeuralNetwork), ct.POINTER(ct.c_double)]
    c_lib.inputDataToNeuralNetwork.restype = ct.POINTER(ct.c_double)
    c_lib.runTests.argtypes = []
    c_lib.runTests.restype = ct.c_int
    c_lib.setRngSeed.argtypes = [ct.c_uint]
    c_lib.setRngSeed.restype = None
    c_lib.getRngSeed.argtypes = []
    c_lib.getRngSeed.restype = ct.c_int

# Defines initialization logic
def _initialize():
    os_type: str = platform.system()
    global c_lib 
    if os_type == "Darwin":
        c_lib = _find_lib("dylib")
        _initialize_lib(c_lib)
        _verify_lib(c_lib)
        
    elif os_type == "Windows":
        c_lib = ct.CDLL = _find_lib("dll")
        _initialize_lib(c_lib)
        _verify_lib(c_lib)
        
    else :
        raise OSError(f"Unsupported OS: {os_type}")
    
# Call the initialization function when the module is imported
_initialize()
