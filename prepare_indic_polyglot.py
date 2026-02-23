import os

def process_file(filename, tag, outfile):
    print(f"Processing {filename} with tag {tag}...")
    if not os.path.exists(filename):
        print(f"  WARNING: {filename} not found. Skipping.")
        return 0
        
    count = 0
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2: continue
            
            src = parts[0].strip()
            trg = parts[1].strip()
            
            # Prepend Language Tag
            # Example: <ENG>wolves or <HIN>khaata
            new_src = f"<{tag}>{src}"
            
            outfile.write(f"{new_src}\t{trg}\n")
            count += 1
    return count

output_file = "multi_indic.train"

with open(output_file, 'w', encoding='utf-8') as f_out:
    print(f"Creating merged dataset: {output_file}...\n")
    
    # 1. English (Morphological Segmentation)
    # Assumes you have 'eng.train' in the folder
    n_eng = process_file("eng.train", "ENG", f_out)
    
    # 2. Indian Languages (Root Recovery)
    # Assumes you ran fetch_all_indic.py and have these files
    n_hin = process_file("hindi.train", "HIN", f_out)
    n_ben = process_file("bengali.train", "BEN", f_out)
    n_kan = process_file("kannada.train", "KAN", f_out)
    n_tel = process_file("telugu.train", "TEL", f_out)
    n_urd = process_file("urdu.train", "URD", f_out)

total = n_eng + n_hin + n_ben + n_kan + n_tel + n_urd
print("-" * 30)
print(f"Success! Created '{output_file}'")
print(f"Total Dataset Size: {total} examples.")