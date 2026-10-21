import os
import glob

# The directory where the new tensors are stored
DATASET_DIR = "/home/l/lialiosk/VoxEffects/training_targets"

# The exact list of authorized DAPS environments
ALLOWED_ENVIRONMENTS = [
    "clean",
    "ipad_balcony1",
    "ipad_bedroom1",
    "ipad_confroom1",
    "ipad_confroom2",
    "ipad_livingroom1",
    "ipad_office1",
    "ipad_office2",
    "ipadflat_confroom1",
    "ipadflat_office1",
    "iphone_balcony1",
    "iphone_bedroom1",
    "iphone_livingroom1"
]

def verify_dataset():
    # Grab all .pt files in both train and val directories
    all_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.pt"), recursive=True)
    
    if not all_files:
        print(f"No .pt files found in {DATASET_DIR}. Is the path correct?")
        return
        
    unauthorized_files = []
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        # Check if any allowed environment tag exists in the filename
        is_valid = any(env in filename for env in ALLOWED_ENVIRONMENTS)
        
        if not is_valid:
            unauthorized_files.append(filename)
            
    print(f"Total .pt files checked: {len(all_files)}")
    
    if unauthorized_files:
        print(f"\n❌ WARNING: Found {len(unauthorized_files)} unauthorized files:")
        for f in unauthorized_files[:20]:
            print(f"  - {f}")
        if len(unauthorized_files) > 20:
            print(f"  ... and {len(unauthorized_files) - 20} more.")
    else:
        print("\n✅ SUCCESS: All generated .pt files strictly belong to the allowed environments!")

if __name__ == "__main__":
    verify_dataset()
