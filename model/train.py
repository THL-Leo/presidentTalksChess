import os
from ultralytics import YOLO

def train_yolo_model():
    # Get current directory where train.py is located
    current_dir = os.getcwd()
    
    # Paths for model, data, and output
    model_path = os.path.join(current_dir, "model", "best.pt")  # Use yolo11n.pt in the same directory
    # model_path = os.path.join(current_dir, "runs", "detect", "train6", "weights", "best.pt")
    data_yaml_path = os.path.join(current_dir, "model", "data.yaml")  # Use data.yaml in the same directory
    output_path = os.path.join(current_dir, "model", "detection_result.jpg")
    test_image_path = os.path.join(current_dir, "data1", "test", "images", "3b1b2-3Q1K1P-2p5-3P1k1Q-P2r4-2P1N3-4P3-6Q1.jpeg")
    
    # Verify files exist
    if not os.path.exists(data_yaml_path):
        print(f"Error: data.yaml not found at {data_yaml_path}")
        return
    
    # Load and train model
    model = YOLO(model_path)
    results = model.train(data=data_yaml_path, epochs=100, imgsz=400, device='mps', batch=32, half=True, int8=True, optimizer='Adam')
    
    # Run inference
    if os.path.exists(test_image_path):
        results = model(test_image_path)
        
        # Save results
        results[0].save(filename=output_path)
        print(f"Image saved to: {output_path}")
        
        # Confirm save worked
        if os.path.exists(output_path):
            print(f"Confirmed: File exists at {output_path}")
            print(f"File size: {os.path.getsize(output_path)} bytes")
        else:
            print(f"Error: File was not created at {output_path}")
    else:
        print(f"Error: Test image not found at {test_image_path}")

# Run the function
if __name__ == "__main__":
    train_yolo_model()