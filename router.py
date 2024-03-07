from absl import app
import sys

def main():
  # gpus = tf.config.list_physical_devices('GPU')
  # if gpus:
  #   try:
  #     # Currently, memory growth needs to be the same across GPUs
  #     for gpu in gpus:
  #       tf.config.experimental.set_memory_growth(gpu, True)
  #     logical_gpus = tf.config.list_logical_devices('GPU')
  #     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  #   except RuntimeError as e:
  #     # Memory growth must be set before GPUs have been initialized
  #     print(e)
    
    script = sys.argv[1]
    
    # Decide which script to call based on the flag
    if script == 'finetune':
        from code.pretrain import run_script1
        run_script1.run()
    elif script == 'pretrain':
        from code.pretrain.run import main
        # run.main(sys.argv)
        app.run(main)
    else:
        print(f'Invalid flag: {script}. Please provide ["finetune", "pretrain", "inference", "visualize"].')

if __name__ == "__main__":
    main()
    