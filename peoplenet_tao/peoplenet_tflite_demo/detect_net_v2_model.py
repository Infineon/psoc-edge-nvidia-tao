"""
DetectNetV2 model for object detection.

This module provides a TensorFlow Lite wrapper for the DetectNetV2 model,
with support for the Ethos-U accelerator.
"""

import os
import sys
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
import logging
from rich.logging import RichHandler
import traceback
from pathlib import Path

# Configure logger
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger("detectnet_v2")

# Import TensorFlow Lite with fallback to tflite_runtime
try:
    import tensorflow.lite as tflite
    logger.debug("Using TensorFlow Lite from main TensorFlow package")
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        logger.debug("Using TensorFlow Lite Runtime")
    except ImportError:
        logger.critical("Failed to import TensorFlow Lite. Please install either tensorflow or tflite_runtime")
        sys.exit(1)


class DetectNetV2Model:
    """
    DetectNetV2 model wrapper for TensorFlow Lite.
    
    This class provides an interface to a TensorFlow Lite model implementing
    the DetectNetV2 architecture for object detection, with optional support
    for the Ethos-U NPU acceleration.
    
    Attributes:
        interpreter (tflite.Interpreter): The TensorFlow Lite interpreter.
        input_index (int): Input tensor index.
        output_0_index (int): Output tensor index for coverage/confidence scores.
        output_1_index (int): Output tensor index for bounding box coordinates.
        using_ethos_u (bool): Whether Ethos-U acceleration is being used.
    """

    def __init__(self, 
                 model_path: str, 
                 num_threads: Optional[int] = None,
                 ethos_u_path: str = "/usr/lib/libethosu_delegate.so") -> None:
        """
        Initialize the DetectNetV2 model with a TensorFlow Lite model file.
        
        Args:
            model_path (str): Path to the TensorFlow Lite model file.
            num_threads (Optional[int], optional): Number of threads to use for inference.
                Defaults to the number of CPU cores available.
            ethos_u_path (str, optional): Path to the Ethos-U delegate library.
                Defaults to "/usr/lib/libethosu_delegate.so".
                
        Raises:
            FileNotFoundError: If the model file doesn't exist.
            RuntimeError: If model initialization fails.
        """
        # Validate model path
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Use all CPU cores if not specified
        if num_threads is None:
            num_threads = os.cpu_count() or 1
            
        self.using_ethos_u = False
        logger.info(f"Initializing DetectNetV2 model from [bold blue]{model_path}[/bold blue] "
                   f"with [bold]{num_threads}[/bold] threads")
        
        delegates = []
        
        try:
            # Removed the Progress display to avoid conflict with other displays
            logger.info("Loading DetectNetV2 model...")
                
            # Try to load the model with and without Ethos-U acceleration
            for attempt in range(2):
                try:
                    # Initialize the TensorFlow Lite interpreter
                    self.interpreter = tflite.Interpreter(
                        model_path=model_path,
                        num_threads=num_threads,
                        experimental_delegates=delegates,
                    )
                    self.interpreter.allocate_tensors()
                    
                    if len(delegates) > 0:
                        self.using_ethos_u = True
                        logger.info("[bold green]Using Ethos-U NPU acceleration[/bold green]")
                    
                    break  # Successfully loaded the model
                    
                except RuntimeError as re:
                    error_msg = str(re)
                    # Check if the error is related to the Ethos-U custom op
                    if (len(delegates) == 0 and 
                        "Encountered unresolved custom op: ethos-u." in error_msg):
                        
                        # Check if Ethos-U delegate exists
                        if not os.path.exists(ethos_u_path):
                            logger.warning(f"Ethos-U delegate library not found at {ethos_u_path}")
                            logger.warning("Continuing without Ethos-U acceleration")
                            continue
                            
                        logger.info("Detected Ethos-U operations, retrying with NPU acceleration")
                        
                        # Retry with the Ethos-U delegate
                        delegates = [tflite.load_delegate(ethos_u_path)]
                        continue
                        
                    # Other error, raise it
                    logger.error(f"Failed to initialize model: {error_msg}")
                    raise re
                    
            # Get input and output tensor details
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            # Map tensor indices
            if len(input_details) != 1:
                logger.warning(f"Expected 1 input tensor, found {len(input_details)}")
                
            if len(output_details) != 2:
                logger.warning(f"Expected 2 output tensors, found {len(output_details)}")
            
            # Store tensor indices
            self.input_index = input_details[0]["index"]
            self.output_0_index = output_details[0]["index"]
            self.output_1_index = output_details[1]["index"]
            
            # Log model details
            input_shape = input_details[0]["shape"]
            input_type = input_details[0]["dtype"]
            logger.debug(f"Model loaded: input shape={input_shape}, type={input_type}")
            logger.debug(f"Output shapes: "
                       f"{output_details[0]['shape']}, {output_details[1]['shape']}")
            
            logger.info("[bold green]✓[/bold green] DetectNetV2 model loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize DetectNetV2 model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"DetectNetV2 model initialization failed: {str(e)}") from e

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference with the DetectNetV2 model.
        
        Args:
            x (np.ndarray): Input image tensor with shape [height, width, channels].
                Should be in RGB format with values in range 0-255.
                
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - Coverage/confidence tensor
                - Bounding box tensor
                
        Raises:
            ValueError: If input tensor has invalid shape or values.
            RuntimeError: If inference fails.
        """
        # Validate input
        if not isinstance(x, np.ndarray):
            raise ValueError("Input must be a numpy array")
            
        if len(x.shape) != 3:
            raise ValueError(f"Input must have 3 dimensions [height, width, channels], "
                           f"got shape {x.shape}")
                           
        if x.shape[2] != 3:
            raise ValueError(f"Input must have 3 channels (RGB), got {x.shape[2]} channels")
            
        try:
            # Preprocess input: normalize to int8 range [-128, 127]
            x = (x - 128).astype(np.int8)
            
            # Add batch dimension
            x = np.expand_dims(x, axis=0)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_index, x)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensors
            cov = self.interpreter.get_tensor(self.output_0_index)
            bbox = self.interpreter.get_tensor(self.output_1_index)
            
            logger.debug(f"Inference completed: cov shape={cov.shape}, bbox shape={bbox.shape}")
            
            return cov, bbox
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Error during model inference: {str(e)}") from e