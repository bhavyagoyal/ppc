import os
import json
import sys

root_folder = sys.argv[1]
all_json_data = []
for root, dirs, files in os.walk(root_folder):
    dirs.sort()
    files.sort()
    for file in files:
        if(file=='scalars.json'):
            json_file = os.path.join(root, file)
            print(json_file)
            with open(json_file,'r') as f:
                all_json_data.append(f.read().strip())
            with open(os.path.join(root, 'scalars_consolidated.json'),'w') as f:
                f.write('\n'.join(all_json_data))

