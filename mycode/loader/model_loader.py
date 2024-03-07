import os
import tensorflow_hub as hub
from set_path import get_proj_path


# @param name: [simclr, remedis]
# @param type: [hub, kereas]
def load_model(name, path, loader="hub", show_summary=False):
    
    #_____________LOAD MODEL______________
    model_path = os.path.join(get_proj_path(), path)
    model = None
    try:
        if loader == "hub":
            model = hub.load(model_path)
    except:
        print(f"The model '{name}' did not load. Please verify the model path. It is also worth considering that the model might still be in the process of being uploaded to the designated location. If you have recently uploaded it to a notebook, there could be delays associated with the upload.")
        raise
    
    # if 'remedis' in name:
    #     model = tf.saved_model.load(model_path, tags=['serve'])
    # elif 'simclr' in name:
    #     model = tf.saved_model.load(model_path, tags=[]).signatures['default']
    # else:
    #     assert False, 'seems like model is not supported'
    
    if show_summary:
        print(f"\nSummary of the loaded model '{name}':")
        if loader == "hub":
            print("model:", model)
            print("tensorflow_version:", model.tensorflow_version)
            print("dir:", dir(model)) # outputs all available functions/props to call
            print("signatures:", list(model.signatures.keys()))
            # print("keras api", model.keras_api)
            # print(f"trainable_variables: (last 10 / {len(model.trainable_variables)})")
            # for variable in model.trainable_variables[-10:]:
            #     print("\t", variable.name)
            # print("call",model.call_and_return_all_conditional_losses)
            # print("regularization_losses:", model.regularization_losses)
            
        elif loader == "keras":
            model.summary()
    
    return model
