import os
import tensorflow as tf
import tensorflow_hub as hub
from set_path import get_proj_path


# @param name: [simclr, remedis]
# @param type: [hub, keras]
def load_model(name, path, loader="hub", input_shape=None, show_summary=False):
    
    #_____________LOAD MODEL______________
    model_path = os.path.join(get_proj_path(), path)
    model = None
    try:
        if loader == "hub":
            model = hub.load(model_path)
        elif loader == "keras":
            model = hub.KerasLayer(
                model_path,
                name = name,
                trainable = False,
                input_shape = [] if None in input_shape else input_shape
                )
        
    except:
        print(f"The model '{name}' did not load successfully. Check compatibility.")
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
            print("model:", model)
            print("name:", model.name)
            print("dir:", dir(model)) # outputs all available functions/props to call
            print("Layer Configuration:", model.get_config())
            print("Specs:", model.input_spec)
            # print("Layers:", model.layers)
            # print("Input Shape:", model.input_shape)
            # print("Output Shape:", model.output_shape)
            # layer = tf.keras.Sequential([model])
            # layer.summary()
    
    return model
