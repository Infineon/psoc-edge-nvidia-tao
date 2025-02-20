"""
Post-processing module for object detection models.

This module handles the transformation of raw model outputs into properly
formatted bounding box coordinates.
"""

import numpy as np
from typing import Tuple, Union, List
import logging
from rich.logging import RichHandler
import traceback

# Configure logger
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger("post_processing")


class PostProcessingModel:
    """
    Post-processes raw model outputs to generate final bounding box coordinates.
    
    This class handles the transformation of raw model outputs, particularly
    coverages (confidence scores) and bounding box coordinates, into properly
    formatted bounding boxes on the original image coordinates.
    
    Attributes:
        stride (int): The stride factor used in the model.
        scale (float): The scale factor to apply to bounding box coordinates.
        offset (float): The offset to apply to the center coordinates.
        centers_x (np.ndarray): Pre-computed x-coordinates for bounding box centers.
        centers_y (np.ndarray): Pre-computed y-coordinates for bounding box centers.
    """

    def __init__(self, 
                 input_shape: Tuple[int, int], 
                 stride: int = 16, 
                 scale: float = 35.0, 
                 offset: float = 0.5) -> None:
        """
        Initialize the post-processing model.
        
        Args:
            input_shape (Tuple[int, int]): The shape of the feature map (height, width).
            stride (int, optional): The stride factor. Defaults to 16.
            scale (float, optional): The scale factor for coordinates. Defaults to 35.0.
            offset (float, optional): The offset for center coordinates. Defaults to 0.5.
            
        Raises:
            ValueError: If input_shape is invalid or other parameters are out of range.
        """
        # Validate inputs
        if not isinstance(input_shape, tuple) or len(input_shape) != 2:
            raise ValueError(f"input_shape must be a tuple of 2 integers, got {input_shape}")
        if input_shape[0] <= 0 or input_shape[1] <= 0:
            raise ValueError(f"input_shape dimensions must be positive, got {input_shape}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        
        logger.debug(f"Initializing PostProcessingModel with input_shape={input_shape}, "
                    f"stride={stride}, scale={scale}, offset={offset}")
        
        self.stride = stride
        self.scale = scale
        self.offset = offset

        # Pre-compute center coordinates for efficiency
        try:
            self.centers_x = np.arange(input_shape[1]) * stride + offset
            self.centers_y = np.arange(input_shape[0]) * stride + offset

            # Reshape for broadcasting
            self.centers_x = self.centers_x[:, np.newaxis]
            self.centers_y = self.centers_y[:, np.newaxis, np.newaxis]
            
            logger.debug(f"Center coordinates computed successfully: "
                        f"x-shape={self.centers_x.shape}, y-shape={self.centers_y.shape}")
        except Exception as e:
            logger.error(f"Failed to compute center coordinates: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Center coordinates computation failed: {str(e)}") from e

    def predict(self, 
                cov: np.ndarray, 
                bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply post-processing to raw model outputs.
        
        Takes coverage (confidence) and bounding box tensors from the model 
        and applies post-processing to obtain final bounding box coordinates.
        
        Args:
            cov (np.ndarray): Coverage/confidence tensor from model output.
                Shape: [batch_size, height, width, num_classes]
            bbox (np.ndarray): Raw bounding box tensor from model output.
                Shape: [batch_size, height, width, num_classes*4]
                
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - Reshaped coverage tensor [batch_size, height*width, num_classes]
                - Processed bounding boxes [batch_size, height*width, num_classes, 4]
                  where each box is [x1, y1, x2, y2]
                
        Raises:
            ValueError: If input tensors have incompatible shapes.
            RuntimeError: If processing fails due to numerical errors.
        """
        # Validate inputs
        if not isinstance(cov, np.ndarray) or not isinstance(bbox, np.ndarray):
            raise ValueError("Coverage and bounding box inputs must be numpy arrays")
        
        if len(cov.shape) != 4:
            raise ValueError(f"Coverage tensor must have 4 dimensions, got {len(cov.shape)}")
        
        expected_bbox_dim = cov.shape[3] * 4
        if bbox.shape[3] != expected_bbox_dim and bbox.shape[3] != cov.shape[3] * 4:
            raise ValueError(f"Bounding box tensor has incompatible shape. "
                           f"Expected last dimension to be {expected_bbox_dim}, got {bbox.shape[3]}")
            
        logger.debug(f"Processing tensors: cov shape={cov.shape}, bbox shape={bbox.shape}")
        
        try:
            # Reshape bounding box tensor to separate the coordinates
            bbox = np.reshape(
                bbox, [cov.shape[0], cov.shape[1], cov.shape[2], cov.shape[3], 4]
            )
            
            # Transform predicted box deltas to absolute coordinates
            # Box format: [x1, y1, x2, y2]
            bbox[:, :, :, :, 0] = self.centers_x - self.scale * bbox[:, :, :, :, 0]  # x1
            bbox[:, :, :, :, 1] = self.centers_y - self.scale * bbox[:, :, :, :, 1]  # y1
            bbox[:, :, :, :, 2] = self.centers_x + self.scale * bbox[:, :, :, :, 2]  # x2
            bbox[:, :, :, :, 3] = self.centers_y + self.scale * bbox[:, :, :, :, 3]  # y2
            
            # Flatten spatial dimensions
            cov = np.reshape(cov, [cov.shape[0], cov.shape[1] * cov.shape[2], cov.shape[3]])
            bbox = np.reshape(
                bbox,
                [
                    bbox.shape[0],
                    bbox.shape[1] * bbox.shape[2],
                    bbox.shape[3],
                    bbox.shape[4],
                ],
            )
            
            # Validate output
            assert cov.shape[0] == bbox.shape[0], "Batch size mismatch in outputs"
            assert cov.shape[1] == bbox.shape[1], "Flattened spatial dimensions mismatch"
            assert cov.shape[2] == bbox.shape[2], "Number of classes mismatch"
            
            logger.debug(f"Post-processing completed successfully. "
                        f"Output shapes: cov={cov.shape}, bbox={bbox.shape}")
            
            return cov, bbox
            
        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Error during bounding box post-processing: {str(e)}") from e