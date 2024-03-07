BATCH_SIZE = 32 # placeholder

models_metadata = {
  'remedis_cxr': {
    'input': (BATCH_SIZE, 224, 224, 3),
    'output': (BATCH_SIZE, 7, 7, 2048)
  },
  'remedis_path': {
    'input': (BATCH_SIZE, 448, 448, 3),
  }
}
