from rembg import remove
from PIL import Image

# Your current image with the white background
input_path = 't20_logo.png' 

# The new perfect transparent image it will create
output_path = 't20_logo_transparent.png'

print("Removing background... please wait.")
input_img = Image.open(input_path)
output_img = remove(input_img)
output_img.save(output_path)
print(f"Success! Saved perfect transparent logo as {output_path}")