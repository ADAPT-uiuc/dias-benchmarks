import sys
import os

url = sys.argv[1]
kernel_slug = url.split('/')[-1]
username = url.split('/')[-2]

if os.path.exists(f"notebooks/{username}/{kernel_slug}"):
  print("EXISTS!")
else:
  print("DOESN'T EXIST.")