import json
import copy

with open('./Final_VRE_Light_sentences_test.json', 'rb') as f:
    annotations = json.load(f)

new_ann = {}

for image_json in annotations['images']:
     new_ann[image_json['filename']] = image_json['reference_pairs'][0]['references']
'''

with open('./obj_det_res.json', 'rb') as f:
    annotations = json.load(f)

anno2 = copy.deepcopy(annotations)
new_ann = {}

for i,obj in enumerate(annotations['images']):
    image_name = obj['file_name'].split('/')[3]
    new_ann[image_name] = []
    for j,ann in enumerate(obj['annotations']):
        new_arr = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3]]
        new_arr2 = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][2]-ann['bbox'][0], ann['bbox'][3]-ann['bbox'][1]]
        new_ann[image_name].append(new_arr)
        anno2['images'][i]['annotations'][j]['bbox'] = new_arr2


with open('./new_bb.json', 'w') as f:
    json.dump(new_ann, f)

with open('./new_obj_det_res.json', 'w') as f:
    json.dump(anno2, f)
'''

with open('ref_annotations.json', 'w') as f:
    json.dump(new_ann, f)
