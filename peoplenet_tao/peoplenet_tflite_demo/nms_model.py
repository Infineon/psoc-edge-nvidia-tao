"""
Non-Maximum Suppression (NMS) module for object detection.

This module provides a TensorFlow Lite model wrapper for performing
Non-Maximum Suppression on detection results to filter overlapping boxes.
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
logger = logging.getLogger("nms_model")

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


class NMSModel:
    """
    Non-Maximum Suppression (NMS) model wrapper for TensorFlow Lite.
    
    This class provides an interface to a TensorFlow Lite model that performs
    Non-Maximum Suppression to filter overlapping detection boxes based on
    their confidence scores and Intersection over Union (IoU).
    
    Attributes:
        interpreter (tflite.Interpreter): The TensorFlow Lite interpreter.
        input_bbox_index (int): Input tensor index for bounding boxes.
        input_cov_index (int): Input tensor index for coverage/confidence scores.
        input_max_output_size_index (int): Input tensor index for max output size.
        input_iou_threshold_index (int): Input tensor index for IoU threshold.
        input_score_threshold_index (int): Input tensor index for score threshold.
        output_bbox_index (int): Output tensor index for filtered bounding boxes.
        output_cov_index (int): Output tensor index for filtered confidence scores.
    """

    def __init__(self, model_path: str, num_threads: Optional[int] = None) -> None:
        """
        Initialize the NMS model with a TensorFlow Lite model file.
        
        Args:
            model_path (str): Path to the TensorFlow Lite model file.
            num_threads (Optional[int], optional): Number of threads to use for inference.
                Defaults to the number of CPU cores available.
                
        Raises:
            FileNotFoundError: If the model file doesn't exist.
            RuntimeError: If model initialization fails.
        """
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Use all CPU cores if not specified
        if num_threads is None:
            num_threads = os.cpu_count() or 1
            
        logger.info(f"Initializing NMS model from [bold blue]{model_path}[/bold blue] "
                   f"with [bold]{num_threads}[/bold] threads")
            
        try:
            # Removed the Progress display to avoid conflict
            logger.info("Loading NMS model...")
            
            # Initialize the TensorFlow Lite interpreter
            self.interpreter = tflite.Interpreter(
                model_path=model_path,
                num_threads=num_threads,
            )
            self.interpreter.allocate_tensors()
            
            # Get input and output tensor details
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            logger.debug(f"Model initialized: {len(input_details)} inputs, "
                        f"{len(output_details)} outputs")
            
            # Map input tensor indices based on tensor names
            self._map_input_indices(input_details)
            
            # Map output tensor indices based on tensor shapes
            self._map_output_indices(output_details)
            
            logger.info("[bold green]✓[/bold green] NMS model loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize NMS model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"NMS model initialization failed: {str(e)}") from e

    def _map_input_indices(self, input_details: List[Dict[str, Any]]) -> None:
        """
        Map input tensor indices based on tensor names.
        
        Args:
            input_details (List[Dict[str, Any]]): List of input tensor details.
            
        Raises:
            RuntimeError: If required input tensors are not found.
        """
        # Initialize indices with None to detect missing tensors
        self.input_bbox_index = None
        self.input_cov_index = None
        self.input_max_output_size_index = None
        self.input_iou_threshold_index = None
        self.input_score_threshold_index = None
        
        # Map tensor indices based on names
        for input_detail in input_details:
            input_name = input_detail["name"]
            input_index = input_detail["index"]
            
            if "bbox" in input_name:
                self.input_bbox_index = input_index
            elif "cov" in input_name:
                self.input_cov_index = input_index
            elif "max_output_size" in input_name:
                self.input_max_output_size_index = input_index
            elif "iou_threshold" in input_name:
                self.input_iou_threshold_index = input_index
            elif "score_threshold" in input_name:
                self.input_score_threshold_index = input_index
                
        # Validate all required inputs were found
        missing_inputs = []
        if self.input_bbox_index is None:
            missing_inputs.append("bbox")
        if self.input_cov_index is None:
            missing_inputs.append("cov/scores")
        if self.input_max_output_size_index is None:
            missing_inputs.append("max_output_size")
        if self.input_iou_threshold_index is None:
            missing_inputs.append("iou_threshold")
        if self.input_score_threshold_index is None:
            missing_inputs.append("score_threshold")
            
        if missing_inputs:
            raise RuntimeError(f"Required input tensors not found: {', '.join(missing_inputs)}")
            
        logger.debug("Input tensor indices mapped successfully")

    def _map_output_indices(self, output_details: List[Dict[str, Any]]) -> None:
        """
        Map output tensor indices based on tensor shapes.
        
        Args:
            output_details (List[Dict[str, Any]]): List of output tensor details.
            
        Raises:
            RuntimeError: If required output tensors are not found.
        """
        # Initialize indices with None to detect missing tensors
        self.output_bbox_index = None
        self.output_cov_index = None
        
        # Map tensor indices based on shapes
        for output_detail in output_details:
            output_shape = output_detail["shape"]
            output_index = output_detail["index"]
            
            if np.array_equal(output_shape, [1, 4]):
                self.output_bbox_index = output_index
            elif np.array_equal(output_shape, [1]):
                self.output_cov_index = output_index
                
        # Validate all required outputs were found
        missing_outputs = []
        if self.output_bbox_index is None:
            missing_outputs.append("bounding boxes")
        if self.output_cov_index is None:
            missing_outputs.append("confidence scores")
            
        if missing_outputs:
            raise RuntimeError(f"Required output tensors not found: {', '.join(missing_outputs)}")
            
        logger.debug("Output tensor indices mapped successfully")

    def predict(self,
                scores: np.ndarray,
                boxes: np.ndarray,
                max_output_size: int = 20,
                iou_threshold: float = 0.3,
                score_threshold: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Non-Maximum Suppression to filter detection boxes.
        
        Takes a set of bounding boxes and corresponding confidence scores,
        and applies NMS to filter out overlapping boxes with lower confidence.
        
        Args:
            scores (np.ndarray): Confidence scores for each box. Shape: [num_boxes]
            boxes (np.ndarray): Bounding boxes with coordinates. Shape: [num_boxes, 4]
            max_output_size (int, optional): Maximum number of boxes to keep. Defaults to 20.
            iou_threshold (float, optional): IoU threshold for overlap filtering. Defaults to 0.3.
            score_threshold (float, optional): Minimum confidence threshold. Defaults to 0.2.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - Filtered confidence scores
                - Filtered bounding boxes
                
        Raises:
            ValueError: If input arrays have invalid shapes or values.
            RuntimeError: If inference fails.
        """
        # Validate inputs
        if not isinstance(scores, np.ndarray) or not isinstance(boxes, np.ndarray):
            raise ValueError("Scores and boxes must be numpy arrays")
            
        if len(scores.shape) != 1:
            raise ValueError(f"Scores must be a 1D array, got shape {scores.shape}")
            
        if len(boxes.shape) != 2 or boxes.shape[1] != 4:
            raise ValueError(f"Boxes must have shape [num_boxes, 4], got {boxes.shape}")
            
        if scores.shape[0] != boxes.shape[0]:
            raise ValueError(f"Number of scores ({scores.shape[0]}) must match "
                        f"number of boxes ({boxes.shape[0]})")
                        
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError(f"IoU threshold must be between 0 and 1, got {iou_threshold}")
            
        if not (0.0 <= score_threshold <= 1.0):
            raise ValueError(f"Score threshold must be between 0 and 1, got {score_threshold}")
            
        if max_output_size <= 0:
            raise ValueError(f"Max output size must be positive, got {max_output_size}")
            
        logger.debug(f"Running NMS with {len(scores)} boxes, max_output_size={max_output_size}, "
                    f"iou_threshold={iou_threshold:.2f}, score_threshold={score_threshold:.2f}")
            
        try:
            self.interpreter.set_tensor(self.input_bbox_index, boxes)
            self.interpreter.set_tensor(self.input_cov_index, scores)
            self.interpreter.set_tensor(
                self.input_max_output_size_index,
                np.array([max_output_size], dtype=np.int32),
            )
            self.interpreter.set_tensor(
                self.input_iou_threshold_index, np.array([iou_threshold], dtype=np.float32)
            )
            self.interpreter.set_tensor(
                self.input_score_threshold_index,
                np.array([score_threshold], dtype=np.float32),
            )

            self.interpreter.invoke()

            boxes = self.interpreter.get_tensor(self.output_bbox_index)
            scores = self.interpreter.get_tensor(self.output_cov_index)

            return scores, boxes
                        
        except Exception as e:
            logger.error(f"NMS inference failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Error during NMS inference: {str(e)}") from e