#!/usr/bin/env python3
import joblib
import numpy as np

model = joblib.load('/home/hosodalab2/Desktop/MyRobot/MyRobot_RasPi_Desktop_Mix/data/MixAll.seed42.preview.smalltrain.step06/20250927T155010/trained_model.joblib')
print('=== Model structure ===')
print('Model type:', type(model))
adapter = getattr(model, 'adapter', None)
print('Has adapter:', adapter is not None)
if adapter:
    params = getattr(adapter, 'params', None)
    print('\n=== Adapter params ===')
    print('Params type:', type(params))
    if params:
        print('Active joints:', getattr(params, 'active_joints', 'NOT FOUND'))
        print('DOF:', getattr(params, 'dof', 'NOT FOUND'))
        print('angle_unit:', getattr(params, 'angle_unit', 'NOT FOUND'))
        print('include_tension:', getattr(params, 'include_tension', 'NOT FOUND'))
        print('preview_step:', getattr(params, 'preview_step', 'NOT FOUND'))
        print('dt:', getattr(params, 'dt', 'NOT FOUND'))
    print('\n=== Adapter methods ===')
    print('dir(adapter):', [x for x in dir(adapter) if not x.startswith('_')])

# Test prediction
print('\n=== Test prediction ===')
# Create dummy input matching runtime format from debug log
# [DEBUG MLP loop=1] X shape=(1, 7), X=[-0. 0. 5.00457875 -55.86172161 2.24602192 2.17817487 2.38221504]
X_test = np.array([[-0., 0., 5.00457875, -55.86172161, 2.24602192, 2.17817487, 2.38221504]])
print('Input X_test:', X_test)
try:
    y_pred = model.predict(X_test)
    print('Prediction y_pred:', y_pred)
except Exception as e:
    print('Prediction failed:', e)
