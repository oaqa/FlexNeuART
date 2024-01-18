try:
    import models_wrapper
except Exception as e:
    print('Failed to load the NDRM models (which require additional libraries): ' + str(e))
