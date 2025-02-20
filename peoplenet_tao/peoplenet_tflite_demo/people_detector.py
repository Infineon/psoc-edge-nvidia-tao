#!/usr/bin/env python3
"""
People Detection System using DetectNetV2.

This module implements a robust people detection system using the DetectNetV2
architecture with TensorFlow Lite, handling the entire pipeline from camera
input to visualizing detected objects.

Example usage:
    python people_detector.py --camera 0 --model resnet34_peoplenet_int8_ex.tflite
"""

import os
import sys
import time
import platform
import argparse
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from dataclasses import dataclass, field
import traceback
import signal
import cv2
import numpy as np

# Configure rich logging
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box
import atexit

# Import custom modules
from detect_net_v2_model import DetectNetV2Model
from nms_model import NMSModel
from post_processing_model import PostProcessingModel

# Create console for rich output
console = Console()

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_path=False)]
)
logger = logging.getLogger("people_detector")


@dataclass
class DetectionConfig:
    """
    Configuration for detection parameters.
    
    This dataclass contains all the configurable parameters for the detection
    pipeline, including model paths, thresholds, and visualization settings.
    
    Attributes:
        model_path (str): Path to the DetectNetV2 model file.
        nms_model_path (str): Path to the NMS model file.
        colors (List[Tuple[int, int, int]]): RGB colors for each detected class.
        max_output_sizes (List[int]): Maximum number of detections per class.
        iou_thresholds (List[float]): IoU thresholds for NMS per class.
        score_thresholds (List[float]): Confidence thresholds per class.
        min_height (int): Minimum height for valid detections.
        labels (List[str]): Class labels for visualization.
        input_size (Tuple[int, int]): Model input size (width, height).
        grid_size (Tuple[int, int]): Size of the detection grid (height, width).
        camera_index (int): Camera device index to use.
        camera_width (int): Requested camera width.
        camera_height (int): Requested camera height.
        display_scale (float): Scale factor for display window.
        show_fps (bool): Whether to display FPS counter.
        fullscreen (bool): Whether to run in fullscreen mode.
        output_path (Optional[str]): Path to save video output (None = no saving).
    """
    
    model_path: str = "resnet34_peoplenet_int8_ex.tflite"
    nms_model_path: str = "nms_model.tflite"
    colors: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    )
    max_output_sizes: List[int] = field(default_factory=lambda: [20, 20, 20])
    iou_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    score_thresholds: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.2])
    min_height: int = 20
    labels: List[str] = field(default_factory=lambda: ["Person", "Bag", "Face"])
    input_size: Tuple[int, int] = (960, 540)  # (width, height)
    grid_size: Tuple[int, int] = (34, 60)  # (height, width)
    camera_index: int = 0
    camera_width: int = 1920
    camera_height: int = 1080
    display_scale: float = 1.0
    show_fps: bool = True
    fullscreen: bool = False
    output_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Ensure lists have the same length as labels
        if len(self.colors) != len(self.labels):
            raise ValueError(f"Number of colors ({len(self.colors)}) must match "
                          f"number of labels ({len(self.labels)})")
        
        if len(self.max_output_sizes) != len(self.labels):
            raise ValueError(f"Number of max_output_sizes ({len(self.max_output_sizes)}) must match "
                          f"number of labels ({len(self.labels)})")
        
        if len(self.iou_thresholds) != len(self.labels):
            raise ValueError(f"Number of iou_thresholds ({len(self.iou_thresholds)}) must match "
                          f"number of labels ({len(self.labels)})")
        
        if len(self.score_thresholds) != len(self.labels):
            raise ValueError(f"Number of score_thresholds ({len(self.score_thresholds)}) must match "
                          f"number of labels ({len(self.labels)})")
        
        # Validate threshold ranges
        for i, iou in enumerate(self.iou_thresholds):
            if not (0.0 <= iou <= 1.0):
                raise ValueError(f"IoU threshold for {self.labels[i]} must be between 0 and 1")
        
        for i, score in enumerate(self.score_thresholds):
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Score threshold for {self.labels[i]} must be between 0 and 1")
        
        # Validate camera settings
        if self.camera_index < 0:
            raise ValueError(f"Camera index must be non-negative, got {self.camera_index}")
        
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError(f"Camera dimensions must be positive, got {self.camera_width}x{self.camera_height}")
        
        if self.display_scale <= 0:
            raise ValueError(f"Display scale must be positive, got {self.display_scale}")
    
    def to_rich_table(self) -> Table:
        """Convert configuration to a rich table for display."""
        table = Table(title="Detection Configuration", box=box.SIMPLE)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        # Add basic parameters
        table.add_row("Model Path", self.model_path)
        table.add_row("NMS Model Path", self.nms_model_path)
        table.add_row("Input Size", f"{self.input_size[0]}x{self.input_size[1]}")
        table.add_row("Grid Size", f"{self.grid_size[0]}x{self.grid_size[1]}")
        table.add_row("Camera Index", str(self.camera_index))
        table.add_row("Camera Resolution", f"{self.camera_width}x{self.camera_height}")
        
        # Add class-specific parameters
        for i, label in enumerate(self.labels):
            table.add_row(
                f"{label} Settings",
                f"Score Threshold: {self.score_thresholds[i]:.2f}, "
                f"IoU: {self.iou_thresholds[i]:.2f}, "
                f"Max Detections: {self.max_output_sizes[i]}"
            )
        
        return table


class PeopleDetector:
    """
    Main class for people detection using DetectNetV2.
    
    This class implements the complete detection pipeline, from camera input
    to visualization of detection results.
    
    Attributes:
        config (DetectionConfig): Configuration for detection parameters.
        model (DetectNetV2Model): The main detection model.
        post_processor (PostProcessingModel): Post-processing for detections.
        nms_model (NMSModel): Non-Maximum Suppression model.
        video_writer (Optional[cv2.VideoWriter]): Writer for output video.
        frame_count (int): Number of frames processed.
        start_time (float): Time when processing started.
        is_running (bool): Flag to control the main loop.
    """
    
    def __init__(self, config: DetectionConfig) -> None:
        """
        Initialize the people detector.
        
        Args:
            config (DetectionConfig): Configuration parameters.
            
        Raises:
            ValueError: If configuration is invalid.
            RuntimeError: If initialization fails.
        """
        self.config = config
        self.video_writer = None
        self.frame_count = 0
        self.start_time = 0
        self.is_running = False
        
        # Setup display settings
        self._setup_display()
        
        try:
            # Initialize models
            self._initialize_models()
        except Exception as e:
            logger.error(f"Failed to initialize detector: {str(e)}")
            raise
    
    def _setup_display(self) -> None:
        """
        Setup display environment for OpenCV.
        
        This handles platform-specific display configurations and
        registers signal handlers for clean shutdown.
        """
        # Set environment variables for display
        if platform.system() == "Linux":
            # For Linux, handle X11 display setting
            if "DISPLAY" not in os.environ:
                os.environ["DISPLAY"] = ":0"
                logger.debug("Set DISPLAY environment variable to :0")
            
            # For Qt-based OpenCV builds
            os.environ["QT_QPA_PLATFORM"] = ""
        
        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register clean shutdown function
        atexit.register(self._cleanup)
        
        logger.debug("Display environment and signal handlers configured")
    
    def _signal_handler(self, sig, frame) -> None:
        """Handle termination signals."""
        logger.info("Termination signal received, shutting down...")
        self.is_running = False
    
    def _cleanup(self) -> None:
        """Clean up resources on exit."""
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
            logger.info(f"Video output saved to {self.config.output_path}")
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        logger.debug("Resources cleaned up")
    
    def _initialize_models(self) -> None:
        """
        Initialize all required models for the detection pipeline.
        
        Raises:
            FileNotFoundError: If model files are not found.
            RuntimeError: If model initialization fails.
        """
        try:
            # Initialize all models sequentially without using Progress
            logger.info("[bold green]Initializing models...[/bold green]")
            
            # Initialize main detection model
            logger.info(f"Loading DetectNetV2 model from {self.config.model_path}")
            self.model = DetectNetV2Model(self.config.model_path)
            
            # Initialize post-processing model
            logger.info(f"Initializing post-processing for grid size {self.config.grid_size}")
            self.post_processor = PostProcessingModel(self.config.grid_size)
            
            # Initialize NMS model
            logger.info(f"Loading NMS model from {self.config.nms_model_path}")
            self.nms_model = NMSModel(self.config.nms_model_path)
            
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Model initialization failed: {e}") from e
        
        logger.info("[bold green]✓[/bold green] All models initialized successfully")
    
    def connect_camera(self) -> cv2.VideoCapture:
        """
        Connect to the specified camera device.
        
        Returns:
            cv2.VideoCapture: The camera capture object.
            
        Raises:
            RuntimeError: If camera connection fails.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold green]Connecting to camera {self.config.camera_index}...[/bold green]"),
            transient=True,
        ) as progress:
            progress.add_task("connect", total=None)
            
            # Initialize camera
            cap = cv2.VideoCapture(self.config.camera_index)
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
            
            # Check if camera opened successfully
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera at index {self.config.camera_index}")
        
        # Log camera properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"[bold green]✓[/bold green] Camera connected - "
                  f"Resolution: {actual_width}x{actual_height}, "
                  f"FPS: {fps:.1f}")
        
        return cap
    
    def setup_video_writer(self, frame_width: int, frame_height: int) -> None:
        """
        Set up video writer for saving output.
        
        Args:
            frame_width (int): Width of the output frames.
            frame_height (int): Height of the output frames.
            
        Raises:
            RuntimeError: If video writer initialization fails.
        """
        if self.config.output_path is None:
            return
        
        try:
            # Create output directory if needed
            output_dir = os.path.dirname(os.path.abspath(self.config.output_path))
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine codec
            if platform.system() == "Windows":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            
            # Initialize video writer
            self.video_writer = cv2.VideoWriter(
                self.config.output_path,
                fourcc,
                20.0,  # Target FPS
                (frame_width, frame_height)
            )
            
            logger.info(f"Video output will be saved to {self.config.output_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize video writer: {e}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Video writer initialization failed: {e}") from e
    
    def draw_detection(self, frame: np.ndarray, box: np.ndarray, label: str, color: Tuple[int, int, int]) -> None:
        """Draw detection box and label on frame"""
        x1, y1, x2, y2 = box.astype(np.int32)
        
        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add label with confidence score
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y1 = max(y1, label_size[1])
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - baseline),
            (x1 + label_size[0], y1),
            color,
            cv2.FILLED
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Process a single frame"""
        start_time = time.time()
        
        # Resize frame
        frame = cv2.resize(frame, self.config.input_size)
        frame = np.pad(frame, pad_width=((0, 4), (0, 0), (0, 0)))
        
        # Convert color space
        x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Model predictions
        cov, bbox = self.model.predict(x)
        cov, bbox = self.post_processor.predict(cov, bbox)
        
        # Process detections
        for label_idx in range(cov.shape[-1]):
            nms_scores, nms_boxes = self.nms_model.predict(
                cov[0, :, label_idx],
                bbox[0, :, label_idx, :],
                self.config.max_output_sizes[label_idx],
                self.config.iou_thresholds[label_idx],
                self.config.score_thresholds[label_idx],
            )
            
            for j in range(nms_scores.shape[0]):
                if nms_scores[j] == 0:
                    break
                    
                box = nms_boxes[j, :]
                height = box[3] - box[1]
                
                if height < self.config.min_height:
                    continue
                    
                # Draw detection with label and confidence score
                label = f"{self.config.labels[label_idx]}"
                self.draw_detection(frame, box, label, self.config.colors[label_idx])
        
        fps = 1.0 / (time.time() - start_time)
        return frame, fps
    
    def run(self) -> None:
        """
        Main run loop for the detector.
        
        This method starts the camera, processes frames, and displays results.
        
        Raises:
            RuntimeError: If an error occurs during execution.
        """
        self.is_running = True
        self.frame_count = 0
        self.start_time = time.time()
        
        try:
            # Connect to camera
            camera = self.connect_camera()
            
            # Set up window
            window_name = "DetectNet V2 People Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # Set fullscreen if configured
            if self.config.fullscreen:
                cv2.setWindowProperty(
                    window_name, 
                    cv2.WND_PROP_FULLSCREEN, 
                    cv2.WINDOW_FULLSCREEN
                )
                
            # Get actual frame dimensions for scaling
            ret, frame = camera.read()
            if not ret:
                raise RuntimeError("Failed to read initial frame from camera")
                
            height, width = frame.shape[:2]
            
            # Calculate display size
            display_width = int(width * self.config.display_scale)
            display_height = int(height * self.config.display_scale)
            
            # Resize window
            cv2.resizeWindow(window_name, display_width, display_height)
            
            # Set up video writer if output path is provided
            if self.config.output_path:
                self.setup_video_writer(width, height)
                
            # Show configuration summary
            console.print(self.config.to_rich_table())
            
            logger.info("[bold green]Starting detection loop[/bold green]")
            
            # Main processing loop
            while self.is_running:
                # Read frame from camera
                ret, frame = camera.read()
                if not ret:
                    logger.warning("Failed to read frame from camera, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Process frame
                processed_frame, fps = self.process_frame(frame)
                
                # Write frame to output file if configured
                if self.video_writer is not None:
                    self.video_writer.write(processed_frame)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Check for user exit (q key)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested exit")
                    break
                
                # Update frame counter
                self.frame_count += 1
            
            # Show summary statistics
            elapsed_time = time.time() - self.start_time
            average_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
            
            summary = Table(title="Detection Summary", box=box.SIMPLE)
            summary.add_column("Metric", style="cyan")
            summary.add_column("Value", style="green")
            
            summary.add_row("Total Frames", str(self.frame_count))
            summary.add_row("Elapsed Time", f"{elapsed_time:.2f} seconds")
            summary.add_row("Average FPS", f"{average_fps:.2f}")
            
            console.print(summary)
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            logger.debug(traceback.format_exc())
            raise RuntimeError(f"Execution failed: {e}") from e
            
        finally:
            # Clean up resources
            if 'camera' in locals():
                camera.release()
            
            cv2.destroyAllWindows()
            logger.info("Detection completed")
            
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="People Detection using DetectNetV2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model paths
    parser.add_argument(
        "--model", type=str, default="resnet34_peoplenet_int8_ex.tflite",
        help="Path to the DetectNetV2 model file"
    )
    parser.add_argument(
        "--nms-model", type=str, default="nms_model.tflite",
        help="Path to the NMS model file"
    )
    
    # Camera settings
    parser.add_argument(
        "--camera", type=int, default=1,
        help="Camera device index"
    )
    parser.add_argument(
        "--width", type=int, default=1920,
        help="Camera capture width"
    )
    parser.add_argument(
        "--height", type=int, default=1080,
        help="Camera capture height"
    )
    
    # Display settings
    parser.add_argument(
        "--display-scale", type=float, default=1.0,
        help="Scale factor for display window"
    )
    parser.add_argument(
        "--fullscreen", action="store_true",
        help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--no-fps", action="store_true",
        help="Hide FPS counter"
    )
    
    # Detection settings
    parser.add_argument(
        "--min-height", type=int, default=20,
        help="Minimum height for valid detections"
    )
    parser.add_argument(
        "--person-threshold", type=float, default=0.4,
        help="Score threshold for person detections"
    )
    parser.add_argument(
        "--bag-threshold", type=float, default=0.2,
        help="Score threshold for bag detections"
    )
    parser.add_argument(
        "--face-threshold", type=float, default=0.2,
        help="Score threshold for face detections"
    )
    
    # Output settings
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save output video (optional)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Create configuration from command line arguments
        config = DetectionConfig(
            model_path=args.model,
            nms_model_path=args.nms_model,
            score_thresholds=[args.person_threshold, args.bag_threshold, args.face_threshold],
            min_height=args.min_height,
            camera_index=args.camera,
            camera_width=args.width,
            camera_height=args.height,
            display_scale=args.display_scale,
            show_fps=not args.no_fps,
            fullscreen=args.fullscreen,
            output_path=args.output
        )
        
        # Create and run detector
        detector = PeopleDetector(config)
        detector.run()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"[bold red]Error:[/bold red] {str(e)}")
        if args.verbose:
            logger.debug(traceback.format_exc())
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()