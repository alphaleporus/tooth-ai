import random
import cv2
import os
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from register_dataset import register_final_di_datasets

# --- CONFIG ---
THRESHOLD = 0.2  # Show predictions with > 20% confidence
# --------------

# 1. Register Data
try:
    register_final_di_datasets("data/final-di")
except AssertionError:
    pass

# 2. Setup Config (CRITICAL FIX: Load from output folder)
cfg = get_cfg()
# We load the config SAVED during training to ensure architecture matches perfectly
config_path = "output/rtx4060_48k/config.yaml"
if not os.path.exists(config_path):
    # Fallback if specific run folder is wrong, try to use the workspace one
    print("Warning: Trained config not found, falling back to workspace config...")
    config_path = "workspace/configs/mask_rcnn_1024x512.yaml"

cfg.merge_from_file(config_path)
cfg.MODEL.WEIGHTS = "output/rtx4060_48k/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESHOLD
cfg.MODEL.DEVICE = "cuda"

print(f"Loaded config from: {config_path}")

# 3. Create Predictor
predictor = DefaultPredictor(cfg)

def inspect_model(dataset_name, title):
    print(f"\n--- {title} ---")
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    # Pick a random image
    d = random.sample(dataset_dicts, 1)[0]
    img = cv2.imread(d["file_name"])
    print(f"Predicting on: {os.path.basename(d['file_name'])}")

    # Run Prediction
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    scores = instances.scores.numpy()
    
    # Print Scores
    if len(scores) > 0:
        print(f"Found {len(scores)} detections.")
        print(f"Top Score: {scores.max()*100:.1f}%")
        print(f"Lowest Score: {scores.min()*100:.1f}%")
    else:
        print("!! NO DETECTIONS FOUND !!")

    # Visualize (FIXED: Removed 'alpha' argument)
    v = Visualizer(img[:, :, ::-1],
                   metadata=MetadataCatalog.get(dataset_name), 
                   scale=1.2,
                   instance_mode=ColorMode.IMAGE  # Default mode
    )
    
    # Draw predictions
    out = v.draw_instance_predictions(instances)
    
    # Show result
    cv2.imshow(f"{title}", out.get_image()[:, :, ::-1])
    print("Press any key in the window to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run Tests
print("Test 1: Checking Training Data (Control Group)...")
inspect_model("tooth_train", "TRAINING SET (Control)")

print("Test 2: Checking Test Data (Evaluation)...")
inspect_model("tooth_test", "TEST SET (Unknown)")