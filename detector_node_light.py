# Import timing, computer vision, ROS 2, message, and YOLO dependencies
import time

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics import YOLO


# Define a ROS 2 node
class YoloDetectorNode(Node):
    # Initialize the detector node and all resources required for processing
    def __init__(self):
        super().__init__("yolo_detector")

        # Configure the model path, camera device, and inference parameters
        self.model_path = (
            "/home/nvidia/jetson_yolo/models/"
            "mouse_bottle_yolo11n_exp2.pt"
        )
        self.camera_id = 0
        self.conf_threshold = 0.5
        self.image_size = 640

        # Load the trained YOLO model
        self.get_logger().info(f"Loading model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.get_logger().info(f"Classes: {self.model.names}")

        # Open the camera
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Create publishers for detections and the average frame rate
        self.detection_pub = self.create_publisher(
            Detection2DArray, "/detections", 10
        )
        self.fps_pub = self.create_publisher(
            Float32, "/detector/fps", 10
        )

        # Initialize successful-frame counting and camera-error tracking
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.camera_error_count = 0

        # Schedule frame processing to run again as soon as possible
        self.timer = self.create_timer(0.001, self.process_frame)

        # Report successful initialization and identify the published topics
        self.get_logger().info(
            "YOLO ROS2 node started."
        )
        self.get_logger().info(
            "Publishing: /detections and /detector/fps only."
        )

    # Read, detect, package, and publish one camera frame
    def process_frame(self):
        # Capture one image from the camera
        ret, frame = self.cap.read()

        # Stop the node after repeated reads fail and reset on success
        if not ret:
            self.camera_error_count += 1
            self.get_logger().warning("Failed to read camera frame.")

            if self.camera_error_count >= 30:
                self.get_logger().error("Camera failed 30 times.")
                rclpy.shutdown()
            return

        self.camera_error_count = 0

        # Run YOLO inference
        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.conf_threshold,
            device=0,
            verbose=False,
        )

        # Select the prediction result for the single input frame.
        result = results[0]

        # Create the frame-level detection message and shared timestamp.
        detection_array = Detection2DArray()
        stamp = self.get_clock().now().to_msg()

        detection_array.header.stamp = stamp
        detection_array.header.frame_id = "camera"

        # Convert every YOLO box into a vision_msgs detection entry.
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = map(
                float, box.xyxy[0].tolist()
            )

            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1

            detection = Detection2D()
            detection.header.stamp = stamp
            detection.header.frame_id = "camera"

            detection.bbox.center.position.x = center_x
            detection.bbox.center.position.y = center_y
            detection.bbox.center.theta = 0.0
            detection.bbox.size_x = width
            detection.bbox.size_y = height

            # Attach the predicted class name and score to the detection.
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(class_name)
            hypothesis.hypothesis.score = confidence

            # Add the completed detection to the current frame result.
            detection.results.append(hypothesis)
            detection_array.detections.append(detection)

        # Publish all detections found in the current frame.
        self.detection_pub.publish(detection_array)

        # Calculate the average FPS across all successful frames.
        self.frame_count += 1
        elapsed = time.perf_counter() - self.start_time
        avg_fps = (
            self.frame_count / elapsed
            if elapsed > 0.0
            else 0.0
        )

        # Publish the running average FPS as a Float32 message.
        fps_msg = Float32()
        fps_msg.data = float(avg_fps)
        self.fps_pub.publish(fps_msg)

    # Release the camera before delegating cleanup to the base class.
    def destroy_node(self):
        if hasattr(self, "cap"):
            self.cap.release()
        super().destroy_node()


# Initialize ROS 2, run the node, and guarantee resource cleanup.
def main(args=None):
    rclpy.init(args=args)
    node = None

    # Process callbacks until shutdown or a keyboard interrupt occurs
    try:
        node = YoloDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # Destroy the node and shut down ROS 2
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# Run the node only when this file is executed as the main program.
if __name__ == "__main__":
    main()
