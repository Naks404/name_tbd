import os
import json

def audit_dataset():
    home_dir = os.environ.get("HOME")
    results_dir = os.path.join(home_dir, "VoxEffects", "results")
    
    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found at {results_dir}")
        return

    # 1. Generate Theoretical Expectations
    genders = ['f', 'm']
    speakers = range(1, 11)
    scripts = range(1, 6)
    environments = [
        'clean', 
        'ipad_balcony1', 'ipad_bedroom1', 'ipad_confroom1', 'ipad_confroom2', 
        'ipad_livingroom1', 'ipad_office1', 'ipad_office2', 
        'ipadflat_confroom1', 'ipadflat_office1', 
        'iphone_balcony1', 'iphone_bedroom1', 'iphone_livingroom1'
    ]

    expected_files = set()
    for g in genders:
        for k in speakers:
            for n in scripts:
                for env in environments:
                    filename = f"{g}{k}_script{n}_{env}_metrics.jsonl"
                    expected_files.add(filename)

    # 2. Scan Actual Reality
    actual_files = set([f for f in os.listdir(results_dir) if f.endswith('.jsonl')])

    missing_files = expected_files - actual_files
    unexpected_files = actual_files - expected_files

    # 3. Audit File Integrity
    incomplete_files = {}
    overcomplete_files = {}
    corrupt_files = []
    empty_files = []

    for filename in actual_files:
        filepath = os.path.join(results_dir, filename)
        
        # Check for empty files
        if os.path.getsize(filepath) == 0:
            empty_files.append(filename)
            continue
            
        line_count = 0
        last_line = ""
        
        with open(filepath, 'r') as f:
            for line in f:
                line_count += 1
                last_line = line.strip()
                
        if line_count < 2520:
            incomplete_files[filename] = line_count
        elif line_count > 2520:
            overcomplete_files[filename] = line_count
            
        # Check JSON validity of the last write
        if last_line:
            try:
                json.loads(last_line)
            except json.JSONDecodeError:
                corrupt_files.append(filename)

    # 4. Print the Audit Report
    print("="*60)
    print(" DAPS DATASET AUDIT REPORT")
    print("="*60)
    print(f"Total Expected Files  : {len(expected_files)}")
    print(f"Total Actual Files    : {len(actual_files)}")
    print("-" * 60)

    if missing_files:
        print(f"\n[!] MISSING FILES ({len(missing_files)}):")
        # Print a sample so it doesn't flood the terminal if there are 100+
        for f in list(missing_files):
            print(f"  - {f}")
        if len(missing_files) > 20: print("  ... (list truncated)")
    else:
        print("\n[+] No missing files found.")

    if unexpected_files:
        print(f"\n[!] UNEXPECTED FILES ({len(unexpected_files)}):")
        for f in unexpected_files:
            print(f"  - {f}")
    else:
        print("\n[+] No unexpected/duplicate filenames found.")

    if incomplete_files:
        print(f"\n[!] INCOMPLETE FILES (< 2520 lines):")
        for f, count in incomplete_files.items():
            print(f"  - {f}: {count} lines")
            
    if overcomplete_files:
        print(f"\n[!] OVERCOMPLETE FILES (> 2520 lines):")
        for f, count in overcomplete_files.items():
            print(f"  - {f}: {count} lines")

    if empty_files:
        print(f"\n[!] EMPTY ZERO-BYTE FILES:")
        for f in empty_files:
            print(f"  - {f}")

    if corrupt_files:
        print(f"\n[!] CORRUPT FILES (Truncated JSON string):")
        for f in corrupt_files:
            print(f"  - {f}")

    if not incomplete_files and not overcomplete_files and not empty_files and not corrupt_files:
        print("\n[+] ALL EXISTING FILES ACHIEVED PERFECT 2520 LINE INTEGRITY.")

if __name__ == "__main__":
    audit_dataset()
