from striprtf.striprtf import rtf_to_text
import os

def convert_rtf_to_txt(input_file, output_file):
    """
    Convert an RTF file to plain text.
    Args:
        input_file (str): Path to the RTF file.
        output_file (str): Path to save the plain text file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            rtf_content = file.read()
        plain_text = rtf_to_text(rtf_content)
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(plain_text)
        print(f"Converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error converting {input_file}: {e}")

# List of RTF files to convert
rtf_files = [
    "ALL Letters.rtf",
    "ESSAYS WARREN.rtf",
    "ALL ANNUAL MEETING TRANSCRIPTS.rtf"
]

# Convert each RTF file to a .txt file
for rtf_file in rtf_files:
    txt_file = os.path.splitext(rtf_file)[0] + ".txt"  # Replace .rtf with .txt
    convert_rtf_to_txt(rtf_file, txt_file)