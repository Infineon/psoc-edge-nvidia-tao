#!/usr/bin/env python3
"""
NMS (Non-Maximum Suppression) model builder script.

This script builds a TensorFlow Lite model for performing Non-Maximum
Suppression on object detection outputs. It creates a TensorFlow model
that takes confidence scores and bounding boxes as input, applies NMS,
and outputs filtered boxes and scores.
"""

import os
import sys
import argparse
from typing import Tuple, List, Dict, Any, Optional
import logging
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
import traceback

# Configure logger
FORMAT = "%(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger("nms_builder")

try:
    import tensorflow as tf
    logger.debug("TensorFlow imported successfully")
except ImportError:
    logger.critical("Failed to import TensorFlow. Please install tensorflow package.")
    sys.exit(1)


def build_nms_model(grid_size: Tuple[int, int], 
                   output_path: str = "nms_model.tflite",
                   quantize: bool = False) -> None:
    """
    Build and save a TensorFlow Lite model for Non-Maximum Suppression.
    
    Args:
        grid_size (Tuple[int, int]): Height and width of the detection grid.
        output_path (str, optional): Path to save the TFLite model. Defaults to "nms_model.tflite".
        quantize (bool, optional): Whether to quantize the model. Defaults to False.
        
    Raises:
        RuntimeError: If model building or conversion fails.
    """
    try:
        height, width = grid_size
        num_boxes = height * width
        
        logger.info(f"Building NMS model for grid size {height}x{width} ({num_boxes} boxes)")
        
        # Define input layers
        cov = tf.keras.Input((), batch_size=num_boxes, name="cov")
        bbox = tf.keras.Input((4), batch_size=num_boxes, name="bbox")
        max_output_size = tf.keras.Input(
            (), batch_size=1, name="max_output_size", dtype=tf.int32
        )
        iou_threshold = tf.keras.Input((), batch_size=1, name="iou_threshold")
        score_threshold = tf.keras.Input((), batch_size=1, name="score_threshold")
        
        logger.debug(f"Input tensors defined:")
        logger.debug(f"- cov: shape={cov.shape}, dtype={cov.dtype}")
        logger.debug(f"- bbox: shape={bbox.shape}, dtype={bbox.dtype}")
        logger.debug(f"- max_output_size: shape={max_output_size.shape}, dtype={max_output_size.dtype}")
        logger.debug(f"- iou_threshold: shape={iou_threshold.shape}, dtype={iou_threshold.dtype}")
        logger.debug(f"- score_threshold: shape={score_threshold.shape}, dtype={score_threshold.dtype}")
        
        # Apply Non-Maximum Suppression
        selected_indices, scores = tf.image.non_max_suppression_with_scores(
            boxes=bbox,
            scores=cov,
            max_output_size=max_output_size[0],
            iou_threshold=iou_threshold[0],
            score_threshold=score_threshold[0],
        )
        
        # Gather boxes using indices
        boxes = tf.gather(bbox, selected_indices, name="boxes")
        
        logger.debug(f"Output tensors:")
        logger.debug(f"- scores: shape={scores.shape}, dtype={scores.dtype}")
        logger.debug(f"- boxes: shape={boxes.shape}, dtype={boxes.dtype}")
        
        # Create the model
        nms_model = tf.keras.Model(
            inputs=[cov, bbox, max_output_size, iou_threshold, score_threshold],
            outputs={"scores": scores, "boxes": boxes},
        )
        
        # Show model summary
        logger.info("Model structure:")
        nms_model.summary(print_fn=logger.info)
        
        # Convert to TFLite
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Converting model to TensorFlow Lite...[/bold green]"),
            transient=True,
        ) as progress:
            progress.add_task("convert", total=None)
            
            converter = tf.lite.TFLiteConverter.from_keras_model(nms_model)
            
            # Set optimization flags if quantization is requested
            if quantize:
                logger.info("Applying post-training quantization")
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
            tflite_model = converter.convert()
        
        # Save the model
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)
            
        model_size = os.path.getsize(output_path) / 1024.0
        logger.info(f"[bold green]✓[/bold green] Model saved to [bold blue]{output_path}[/bold blue] "
                   f"({model_size:.2f} KB)")
        
    except Exception as e:
        logger.error(f"Failed to build NMS model: {str(e)}")
        logger.debug(traceback.format_exc())
        raise RuntimeError(f"Error while building NMS model: {str(e)}") from e


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build NMS model for object detection")
    
    parser.add_argument(
        "--height", type=int, default=34,
        help="Height of the detection grid (default: 34)"
    )
    parser.add_argument(
        "--width", type=int, default=60,
        help="Width of the detection grid (default: 60)"
    )
    parser.add_argument(
        "--output", type=str, default="nms_model.tflite",
        help="Output path for the TFLite model (default: nms_model.tflite)"
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="Apply post-training quantization"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    try:
        # Build and save the model
        build_nms_model(
            grid_size=(args.height, args.width),
            output_path=args.output,
            quantize=args.quantize
        )
    except Exception as e:
        logger.error(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()