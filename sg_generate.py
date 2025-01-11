import numpy as np
import random
import h5py
from layout_chatgpt import generate_box_gpt4,draw_box
from collections import deque
import json

with h5py.File('sg_datasets/scene_graph.h5', 'r') as f:
    object_names = [x.decode() for x in f['object_names'][:]]
    relation_names = [x.decode() for x in f['relation_names'][:]]
    relationship_predicates = f['relationship_predicates'][:]


def parse_args():
    parser = argparse.ArgumentParser(description='Train model with specified dataset')
    parser.add_argument('--object', required=True, help='object name')


def find_belonging_to_relations(relation_array, obj_id, relation_id=43):

    relations = np.where(relation_array[obj_id, :, :] == 1)
    triples = []
    
    for target_obj, relation in zip(relations[0], relations[1]):
        if relation == relation_id:  
            triples.append((obj_id, relation, target_obj))
    
    return triples

def generate_random_scene_with_constraints(relation_array, obj_id, relation_id=43, max_depth=3, max_objects=5):

    scene = []  
    visited = set()  
    stack = []  
    

    belonging_to_triples = find_belonging_to_relations(relation_array, obj_id, relation_id)
    

    start_obj_ids = [triple[2] for triple in belonging_to_triples]  
    stack.extend([(obj_id, 0)]) 
    stack.extend([(obj, 0) for obj in start_obj_ids])  


    while stack and len(scene) < max_objects:
        current_obj, depth = random.choice(stack)
        stack.remove((current_obj, depth))
        if current_obj in visited or depth >= max_depth:
            continue
        visited.add(current_obj)


        is_subject =True
        if is_subject:
            relations = np.where(relation_array[current_obj, :, :] == 1)
            if len(relations[0]) > 0:
                idx = random.randint(0, len(relations[0]) - 1)
                target_obj = relations[0][idx]
                relation = relations[1][idx]

                if current_obj in start_obj_ids:
                    current_obj = obj_id
                
                scene.append((current_obj, relation, target_obj))
                stack.append((target_obj, depth + 1))
        else:
            relations = np.where(relation_array[:, current_obj, :] == 1)
            if len(relations[0]) > 0:
                idx = random.randint(0, len(relations[0]) - 1)
                source_obj = relations[0][idx]
                relation = relations[1][idx]
                
                if current_obj in start_obj_ids:
                    current_obj = obj_id
                
                scene.append((source_obj, relation, current_obj))
                stack.append((source_obj, depth + 1))

    return scene


if __name__ == '__main__':
    args = parse_args()
    object_name = f'<new> {args.object}'
    obj_id = object_names.index(object_name)  
    triplets_dict_list = []
    triplets_list_total = []
    # random_scene = generate_random_scene_with_constraints(relationship_predicates, obj_id=obj_id, max_depth=5, max_objects=5)
    for i in range(20):
        random_scene = generate_random_scene_with_constraints(relationship_predicates, obj_id, relation_id=43, max_depth=3, max_objects=1)
        print(f"random scene: {object_names[obj_id]}")
        triplets_list = []
        triplets_dict = {}
        for subject, relation, object_ in random_scene:
            if relation_names[relation] == 'belonging to':
                continue
            # print(f"{object_names[subject]}, {relation_names[relation]}, {object_names[object_]}")
            triplets_list.append(f"{object_names[subject]}, {relation_names[relation]}, {object_names[object_]}")
            triplets_dict['subject'] = object_names[subject]
            triplets_dict['object'] = object_names[object_]
            triplets_dict['relation'] = relation_names[relation]
         
        # print(triplets_list)   
        if triplets_list in triplets_list_total:
            continue
        triplets_list_total.append(triplets_list)
        print(triplets_list_total) 
        if triplets_list !=[]:
            text = f'{triplets_list}'    
            name_objects, boxes_of_object, background_prompt = generate_box_gpt4(text)
            print(name_objects, boxes_of_object)
            

            if name_objects[0] in triplets_dict['subject']:
                triplets_dict['subject_box'] = boxes_of_object[0]
                triplets_dict['object_box'] = boxes_of_object[1]
            else:
                triplets_dict['subject_box'] = boxes_of_object[1]
                triplets_dict['object_box'] = boxes_of_object[0]
            print('background:', background_prompt)
            triplets_dict['background'] = background_prompt
            draw_box(name_objects, boxes_of_object, 'outputs', f'test{i}.jpg')
            triplets_dict_list.append(triplets_dict)

        with open(f"bbox/{args.object}.json", "w", encoding="utf-8") as json_file:
            json.dump(triplets_dict_list, json_file, indent=4, ensure_ascii=False)