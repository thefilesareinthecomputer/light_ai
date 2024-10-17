@staticmethod
def summarize_module(module, class_name=None, method_name=None):
    '''Summarize the classes and methods in a module. Allows drilling down to specific class or method level.'''
    print('### BEGINNING SUMMARIZE MODULE ###')
    summary = {'classes': {}}
    
    # Get all classes in the module that are defined in the current file
    classes = inspect.getmembers(module, inspect.isclass)
    
    if not class_name:
        # Return list of all classes
        return {cls_name: {'docstring': inspect.getdoc(cls_obj)} 
                for cls_name, cls_obj in classes if cls_obj.__module__ == module.__name__}
    
    # If class_name is provided, get selected class details
    selected_class = next((cls_obj for cls_name, cls_obj in classes 
                        if cls_name == class_name and cls_obj.__module__ == module.__name__), None)
    
    if not selected_class:
        return f"Class '{class_name}' not found."
    
    cls_summary = {
        'docstring': inspect.getdoc(selected_class),
        'methods': {}
    }
    
    # Get methods for the class (only from the current file)
    methods = inspect.getmembers(selected_class, inspect.isfunction)
    cls_summary['methods'] = {method_name_: {'docstring': inspect.getdoc(method_obj), 
                                            'source_code': inspect.getsource(method_obj)} 
                            for method_name_, method_obj in methods 
                            if method_obj.__module__ == module.__name__}
    
    summary['classes'][class_name] = cls_summary
    
    if method_name:
        # Get details for a specific method
        selected_method = cls_summary['methods'].get(method_name)
        if not selected_method:
            return f"Method '{method_name}' not found in class '{class_name}'."
        return {method_name: selected_method}
    
    print('### ENDING SUMMARIZE MODULE ###')
    print(f"Summary: \n{summary}")
    return summary