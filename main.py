#!/usr/bin/env python3
"""
Legacy main.py - Simple command-line interface for action recognition
Now uses the optimized ActionRecognitionStream class
"""

import argparse
import os
import sys
import time
import logging
from action_recognition_stream import ActionRecognitionStream, ActionEvent, StreamStatus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEventHandler:
    """Simple event handler for command-line interface"""
    
    def __init__(self, show_enhanced_only: bool = False):
        self.show_enhanced_only = show_enhanced_only
        self.event_count = 0
        self.last_display_time = 0
        
    def on_event(self, event: ActionEvent):
        """Handle action events"""
        self.event_count += 1
        
        # Filter events if requested
        if self.show_enhanced_only and not event.enhanced:
            return
            
        # Throttle display to avoid spam
        current_time = time.time()
        if current_time - self.last_display_time >= 0.5:  # Display every 0.5 seconds
            confidence_str = f"{event.confidence*100:.1f}%"
            enhanced_str = " [ENHANCED]" if event.enhanced else ""
            
            print(f"[{event.timestamp:.1f}s] {event.action_name} ({confidence_str}){enhanced_str}")
            self.last_display_time = current_time

def main(args):
    """Main function for command-line interface"""
    # Input validation
    if not os.path.exists(args.input_video):
        logger.error(f"Input video file not found: {args.input_video}")
        return
    
    # Create event handler
    event_handler = SimpleEventHandler(show_enhanced_only=args.show_enhanced_only)
    
    # Initialize action recognition stream
    try:
        logger.info("Initializing action recognition stream...")
        
        # Convert target actions to config format if provided
        target_actions_override = None
        if args.target_actions:
            target_actions_override = args.target_actions
        
        stream = ActionRecognitionStream(
            config_path=args.config,
            yolo_model_path=args.yolo_model if args.yolo_model != 'yolo11x-pose.pt' else None,
            skateformer_config_path=args.skateformer_config if args.skateformer_config != 'configs/SkateFormer_j.yaml' else None,
            skateformer_weights_path=args.skateformer_weights if args.skateformer_weights != 'pretrained/ntu60_CView/SkateFormer_j.pt' else None,
            target_actions=target_actions_override,
            boost_factor=args.boost_factor if args.boost_factor != 3.0 else None,
            fps_target=args.fps_target if args.fps_target != 30 else None,
            event_callback=event_handler.on_event
        )
        
        # Print configuration summary
        stats = stream.get_stats()
        logger.info("=== CONFIGURATION SUMMARY ===")
        logger.info(f"Dataset: {stats['dataset_type']} ({stats['num_classes']} classes)")
        logger.info(f"Target actions: {len(stats['target_actions'])} actions")
        logger.info(f"Enhanced actions: {list(stats['target_actions_config'].keys())}")
        logger.info(f"Device: {stats['device']}")
        logger.info(f"FPS target: {stats['fps_target']}")
        logger.info(f"Event filtering: {stats['event_filtering']['enabled']}")
        
        # Start stream
        logger.info(f"Starting stream: {args.input_video}")
        if not stream.start_stream(args.input_video):
            logger.error("Failed to start stream")
            return
        
        # Monitor stream
        logger.info("Processing stream... Press Ctrl+C to stop")
        start_time = time.time()
        
        try:
            while True:
                # Check stream status
                status = stream.get_status()
                if status == StreamStatus.ERROR:
                    logger.error("Stream encountered an error")
                    break
                elif status == StreamStatus.STOPPED:
                    logger.info("Stream processing completed")
                    break
                
                # Print periodic stats
                if int(time.time()) % 10 == 0:  # Every 10 seconds
                    stats = stream.get_stats()
                    elapsed = time.time() - start_time
                    fps = stats['frame_count'] / elapsed if elapsed > 0 else 0
                    logger.info(f"Stats: {stats['frame_count']} frames, {event_handler.event_count} events, {fps:.1f} FPS")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Stopping stream...")
        
        # Final stats
        stats = stream.get_stats()
        elapsed = time.time() - start_time
        fps = stats['frame_count'] / elapsed if elapsed > 0 else 0
        
        logger.info("=== FINAL RESULTS ===")
        logger.info(f"Total frames processed: {stats['frame_count']}")
        logger.info(f"Total events generated: {event_handler.event_count}")
        logger.info(f"Processing time: {elapsed:.1f}s")
        logger.info(f"Average FPS: {fps:.1f}")
        
        # Enhanced action summary
        if stats['target_actions']:
            logger.info("=== ENHANCED ACTIONS SUMMARY ===")
            for action_id, action_info in stats['target_actions_config'].items():
                logger.info(f"  {action_id}: {action_info['name']} "
                          f"(priority: {action_info['priority']}, "
                          f"boost: {action_info['boost_factor']:.1f}x)")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return
    
    finally:
        # Cleanup
        if 'stream' in locals():
            stream.stop_stream()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + SkateFormer Action Recognition Stream")
    
    # --- Input Arguments ---
    parser.add_argument('--input-video', type=str, required=True, 
                        help="Path to input video file, webcam index (e.g., 0), or RTSP URL")
    
    # --- Configuration Arguments ---
    parser.add_argument('--config', type=str, default='configs/stream_config.yaml',
                        help="Path to configuration file")
    
    # --- Model Arguments (optional overrides) ---
    parser.add_argument('--yolo-model', type=str, default='yolo11x-pose.pt',
                        help="Path to YOLO pose estimation model")
    parser.add_argument('--skateformer-config', type=str, default='configs/SkateFormer_j.yaml',
                        help="Path to SkateFormer model config file")
    parser.add_argument('--skateformer-weights', type=str, default='pretrained/ntu60_CView/SkateFormer_j.pt',
                        help="Path to SkateFormer model weights")
    
    # --- Action Enhancement Arguments (optional overrides) ---
    parser.add_argument('--target-actions', type=int, nargs='+', 
                        help="List of target action indices to enhance (overrides config)")
    parser.add_argument('--boost-factor', type=float, default=3.0,
                        help="Factor to boost target action probabilities (overrides config)")
    parser.add_argument('--fps-target', type=int, default=30,
                        help="Target FPS for processing (overrides config)")
    
    # --- Display Arguments ---
    parser.add_argument('--show-enhanced-only', action='store_true',
                        help="Only show enhanced target actions")
    
    args = parser.parse_args()
    main(args)
