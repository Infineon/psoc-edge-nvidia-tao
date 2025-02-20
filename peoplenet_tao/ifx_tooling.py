import os
import subprocess
import logging
from typing import Optional, Dict, Union
from pathlib import Path
import onnx
from tflite_converter import convert_to_tflite
import sys
from datetime import datetime

class ModelConversionError(Exception):
    """Custom exception for model conversion pipeline errors."""
    pass

class ONNXProcessor:
    """Class to handle ONNX model processing operations."""
    
    @staticmethod
    def remove_suffix(input_string: str, suffix: str) -> str:
        """
        Remove a specified suffix from a string if it exists.
        
        Args:
            input_string (str): The input string to process
            suffix (str): The suffix to remove
            
        Returns:
            str: The processed string with suffix removed if it existed
        """
        if suffix and input_string.endswith(suffix):
            return input_string[:-len(suffix)]
        return input_string

    @staticmethod
    def remove_suffix_from_onnx(model: onnx.ModelProto, suffix: str = ':0') -> onnx.ModelProto:
        """
        Remove specified suffix from ONNX model input and output names.
        
        Args:
            model (onnx.ModelProto): The input ONNX model
            suffix (str): The suffix to remove from names (default: ':0')
            
        Returns:
            onnx.ModelProto: The processed ONNX model
        """
        graph_input_names = [input.name for input in model.graph.input]
        graph_output_names = [output.name for output in model.graph.output]

        # Process inputs
        for input in model.graph.input:
            input.name = ONNXProcessor.remove_suffix(input.name, suffix)

        # Process outputs
        for output in model.graph.output:
            output.name = ONNXProcessor.remove_suffix(output.name, suffix)

        # Process nodes
        for node in model.graph.node:
            for i in range(len(node.input)):
                if node.input[i] in graph_input_names:
                    node.input[i] = ONNXProcessor.remove_suffix(node.input[i], suffix)

            for i in range(len(node.output)):
                if node.output[i] in graph_output_names:
                    node.output[i] = ONNXProcessor.remove_suffix(node.output[i], suffix)

        return model

def setup_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Set up a custom logger with consistent formatting and optional file output.
    
    Args:
        name (str): Name of the logger
        log_file (Optional[Path]): Path to log file if file logging is desired
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler with custom formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # If log file is specified, create file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def run_ifx_tooling(
    onnx_model_path: Union[str, Path],
    input_shape: list,
    output_dir: Union[str, Path],
    config: Optional[Dict] = None
) -> Dict[str, Path]:
    """
    Execute the IFX tooling pipeline for model conversion:
    ONNX -> OpenVINO -> TensorFlow -> TFLite -> Vela
    
    [Rest of docstring remains the same]
    """
    # Input validation
    if not isinstance(input_shape, list) or len(input_shape) != 4:
        raise ValueError("input_shape must be a list of 4 dimensions [batch, channels, height, width]")
    
    # Setup paths
    output_dir = Path(output_dir)
    onnx_model_path = Path(onnx_model_path)
    
    # Setup logging with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'ifx_tooling_{timestamp}.log'
    
    logger = setup_logger("ifx_tooling", log_file)
    logger.info(f"Starting IFX tooling pipeline")
    logger.info(f"Input model: {onnx_model_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    
    # Initialize configuration
    default_config = {
        'vela_accelerator': 'ethos-u55-128',
        'vela_system_config': 'PSE84_M55_U55_400MHz',
        'vela_memory_mode': 'Sram_Only',
        'compress_to_fp16': False,
        'vela_ini_file_path': 'vela.ini'
    }
    config = {**default_config, **(config or {})}
    logger.info(f"Configuration: {config}")
    
    # Create output directories
    output_paths = {
        'onnx_processed': output_dir / 'processed_onnx' / f"{onnx_model_path.stem}_processed.onnx",
        'openvino': output_dir / 'model_openvino',
        'tensorflow': output_dir / 'model_tf',
        'tflite': output_dir / f"{onnx_model_path.stem}.tflite",
        'vela': output_dir / 'model_vela',
    }
    
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Process ONNX model
        logger.info("🔄 Processing ONNX model...")
        onnx_model = onnx.load(str(onnx_model_path))
        processed_model = ONNXProcessor.remove_suffix_from_onnx(onnx_model)
        onnx.save(processed_model, str(output_paths['onnx_processed']))
        logger.info(f"✅ ONNX processing complete. Output: {output_paths['onnx_processed']}")

        # Step 2: Convert to OpenVINO
        logger.info("🔄 Converting to OpenVINO format...")
        mo_cmd = [
            'mo',
            '--input_model', str(output_paths['onnx_processed']),
            '--input_shape', str(input_shape),
            '--output_dir', str(output_paths['openvino']),
            '--compress_to_fp16', str(config['compress_to_fp16'])
        ]
        result = subprocess.run(mo_cmd, capture_output=True, text=True, check=True)
        logger.info(f"✅ OpenVINO conversion complete. Output: {output_paths['openvino']}")
        
        # Step 3: Convert to TensorFlow
        logger.info("🔄 Converting to TensorFlow format...")
        openvino_model = output_paths['openvino'] / f"{onnx_model_path.stem}_processed.xml"
        if not openvino_model.exists():
            raise FileNotFoundError(f"OpenVINO model not found at: {openvino_model}")
        
        o2tf_cmd = [
            'openvino2tensorflow',
            '--model_path', str(openvino_model),
            '--model_output_path', str(output_paths['tensorflow']),
            '--non_verbose',
            '--output_saved_model'
        ]
        result = subprocess.run(o2tf_cmd, capture_output=True, text=True, check=True)
        logger.info(f"✅ TensorFlow conversion complete. Output: {output_paths['tensorflow']}")

        # Step 4: Convert to TFLite
        logger.info("🔄 Converting to TFLite format...")
        convert_to_tflite(
            str(output_paths['tensorflow']),
            str(output_paths['tflite'])
        )
        logger.info(f"✅ TFLite conversion complete. Output: {output_paths['tflite']}")

        # Step 5: Run Vela compiler
        logger.info("🔄 Running Vela compiler...")
        vela_output_file = output_paths['vela'] / 'vela_output.txt'
        vela_cmd = [
            'vela',
            '--accelerator-config', config['vela_accelerator'],
            '--config', config['vela_ini_file_path'],
            '--system-config', config['vela_system_config'],
            '--memory-mode', config['vela_memory_mode'],
            '--output-dir', str(output_paths['vela']),
            str(output_paths['tflite'])
        ]
        try:
            result = subprocess.run(vela_cmd, capture_output=True, text=True, check=True)
            
            # Save Vela output to file
            with open(vela_output_file, 'w') as f:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
            
            logger.info(f"✅ Vela compilation complete")
            logger.info(f"📝 Vela output saved at: {vela_output_file}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Vela compilation failed:\nOutput: {e.output}\nError: {e.stderr}")
            raise ModelConversionError(f"Vela compilation failed. Please check the error logs above.")

        # Verify outputs
        logger.info("🔍 Verifying outputs...")
        for name, path in output_paths.items():
            if not path.exists():
                raise ModelConversionError(f"Expected output not found: {name} at {path}")

        logger.info("✨ Model conversion pipeline completed successfully")
        return {k: v for k, v in output_paths.items()}

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Command failed: {e.cmd}\nOutput: {e.output}\nError: {e.stderr}")
        raise ModelConversionError(f"Command failed: {e.cmd}\nOutput: {e.output}\nError: {e.stderr}")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise ModelConversionError(f"Pipeline failed: {str(e)}")