import tensorflow as tf
import numpy as np

def representative_dataset():
    """Generate representative dataset for quantization."""
    for _ in range(100):
        yield [
            (np.random.randint(0, 255, [1, 544, 960, 3])).astype(np.float32) / 255.0
        ]

def convert_to_tflite(input_path, output_path):
    """
    Convert TensorFlow SavedModel to quantized TFLite model.
    
    Args:
        input_path (str): Path to input SavedModel directory
        output_path (str): Path to save the output TFLite model
        
    Returns:
        bool: True if conversion successful
    """
    try:
        # Initialize converter
        converter = tf.lite.TFLiteConverter.from_saved_model(input_path)
        
        # Set optimization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = representative_dataset
        
        # Convert the model
        tflite_quant_model = converter.convert()
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_quant_model)
            
        return True
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert TensorFlow SavedModel to quantized TFLite model.')
    parser.add_argument('--input', required=True, type=str, help='Path to the input SavedModel directory')
    parser.add_argument('--output', required=True, type=str, help='Path to save the TFLite model')
    args = parser.parse_args()
    
    convert_to_tflite(args.input, args.output)