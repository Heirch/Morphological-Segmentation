import requests
import os

# Dictionary of Language Code -> UniMorph Raw URL
# We use the "blister" or "master" raw files from UniMorph
LANGUAGES = {
    "Hindi": "https://raw.githubusercontent.com/unimorph/hin/master/hin",
    "Bengali": "https://raw.githubusercontent.com/unimorph/ben/master/ben",
    "Kannada": "https://raw.githubusercontent.com/unimorph/kan/master/kan",
    "Telugu": "https://raw.githubusercontent.com/unimorph/tel/master/tel",
    "Urdu": "https://raw.githubusercontent.com/unimorph/urd/master/urd"
}

def download_lang(lang_name, url):
    output_file = f"{lang_name.lower()}.train"
    print(f"Downloading {lang_name} data...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.text.splitlines()
        count = 0
        
        with open(output_file, "w", encoding="utf-8") as f:
            for line in data:
                if not line.strip(): continue
                parts = line.split('\t')
                if len(parts) < 2: continue
                
                lemma = parts[0].strip()
                form = parts[1].strip()
                
                if lemma == form: continue
                
                # Format: Form -> Lemma (Root Recovery)
                f.write(f"{form}\t{lemma}\n")
                count += 1
                
        print(f"  -> Success! Saved {count} pairs to '{output_file}'")
        return count

    except Exception as e:
        print(f"  -> Failed: {e}")
        return 0

total_count = 0
print("--- STARTING DOWNLOAD ---")
for lang, url in LANGUAGES.items():
    total_count += download_lang(lang, url)

print("-------------------------")
print(f"Total Initial Dataset Size: {total_count} words")