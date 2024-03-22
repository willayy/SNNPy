import platform, os

# Defines initialization logic
def initialize():
    os_type: str = platform.system()
    if os_type == "Darwin":
        shared_lib_path: str = os.path.join(os.path.dirname(__file__), "sharedlib/libSNNPy.dylib")
        if not os.path.isfile(shared_lib_path):
            raise FileNotFoundError(f"Shared library not found at {shared_lib_path}")
    elif os_type == "Windows":
        shared_lib_path: str = os.path.join(os.path.dirname(__file__), "sharedlib/libSNNPy.dll")
        if not os.path.isfile(shared_lib_path):
            raise FileNotFoundError(f"Shared library not found at {shared_lib_path}")
    else :
        raise OSError(f"Unsupported OS: {os_type}")
    
# Call the initialization function when the module is imported
initialize()
