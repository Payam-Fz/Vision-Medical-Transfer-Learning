import zipfile
import tensorflow as tf

def load_model(model_path: str):
    with zipfile.ZipFile(model_path, "r") as zip_ref:
        zip_ref.extractall("Probing_ViTs/")
    model_name = model_path.split(".")[0]
    loaded_model = tf.keras.models.load_model(model_name, compile=False)
    return loaded_model
