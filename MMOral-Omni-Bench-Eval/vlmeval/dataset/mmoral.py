import os
import os.path as osp
import random
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from .utils import build_judge, DEBUG_MESSAGE

from .image_base import ImageBaseDataset
# from ..smp import load, dump, d2df, read_ok, decode_base64_to_image_file, toliststr, get_intermediate_file_path
from ..utils import track_progress_rich
from ..smp import *

def img_root_map(dataset):
    if 'MM_NIAH' in dataset:
        return 'MMNIAH'
    if 'CRPE' in dataset:
        return 'CRPE'
    if 'OCRVQA' in dataset:
        return 'OCRVQA'
    if 'COCO_VAL' == dataset:
        return 'COCO'
    if 'MMMU' in dataset:
        return 'MMMU'
    if "QSpatial" in dataset:
        return "QSpatial"

    mmbench_root_map = {
        'MMBench_DEV_EN': 'MMBench', 'MMBench_TEST_EN': 'MMBench',
        'MMBench_DEV_CN': 'MMBench', 'MMBench_TEST_CN': 'MMBench',
        'MMBench_DEV_KO': 'MMBench',
        'MMBench': 'MMBench', 'MMBench_CN': 'MMBench',
        'MMBench_DEV_EN_V11': 'MMBench_V11', 'MMBench_TEST_EN_V11': 'MMBench_V11',
        'MMBench_DEV_CN_V11': 'MMBench_V11', 'MMBench_TEST_CN_V11': 'MMBench_V11',
        'MMBench_V11': 'MMBench', 'MMBench_CN_V11': 'MMBench',
    }
    if dataset in mmbench_root_map:
        return mmbench_root_map[dataset]
    return dataset

def build_mmoral_gpt4_prompt(line):
    question = line['question']
    gt = str(line['answer'])
    prediction = str(line['prediction'])
    prompt = """
Given the question, compare the ground truth and prediction from AI models, to generate a correctness score for the prediction.
The correctness score is 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right).
Just complete the last space of the correctness score.

Question | Ground truth | Prediction | Correctness
--- | --- | --- | ---
How many teeth are visualized in the radiograph? | 30 teeth are visualized with clear anatomical definition. | 30 | 1.0
How many teeth are visualized in the radiograph? | 30 teeth are visualized with clear anatomical definition. | 29 teeth are visualized with clear anatomical definition. | 0.0
What is the status of the wisdom teeth in the radiograph? | Three wisdom teeth are detected, all of which are impacted: #18, #28, and #48. | #18: impacted, #28: impacted, #48: erupted | 0.7
What is the condition of the teeth #26 and #14? | Teeth #26 and #14 show signs of periapical abscesses. | Teeth #26 and #23 show signs of periapical abscesses. | 0.5
What is the condition of the bone architecture and visible structures in the jaw? | No apparent bone loss is observed. Bilateral mandibular canals and maxillary sinuses are clearly visible. | Bilateral mandibular canals and maxillary sinuses are clearly visible. | 0.5
What is the clinical priority concern regarding the periapical lesions? | Periapical cysts at #11 and #12, and granuloma at #46 require endodontic evaluation. | Periapical lesions at #11, #12, and #46 require endodontic evaluation. | 0.8
What radiographic features are visible in tooth #31 on the panoramic X-ray? | [\n{\"Teeth position\": {\"point_2d\": [1242, 726]}},\n{\"Crown\": {\"box_2d\": [1220, 637, 1266, 741]}}\n] | Crown | 0.8
What radiographic features are visible in tooth #31 on the panoramic X-ray? | [\n{\"Teeth position\": {\"point_2d\": [1242, 726]}},\n{\"Crown\": {\"box_2d\": [1220, 637, 1266, 741]}}\n] | Crown at position: [1230, 627, 1276, 750] | 0.9
What radiographic features are visible in tooth #31 on the panoramic X-ray? | [\n{\"Teeth position\": {\"point_2d\": [1242, 726]}},\n{\"Crown\": {\"box_2d\": [1220, 637, 1266, 741]}}\n] | Teeth at position: {\"point_2d\": [1242, 726]}},\n{Crown at position: {\"box_2d\": [1230, 627, 1276, 750]}} | 1.0
"""
    gpt4_prompt = prompt + '\n' + ' | '.join(
        [question, gt.replace('<AND>', ' <AND> ').replace('<OR>', ' <OR> '), prediction, ''])
    return gpt4_prompt

def MMOral_auxeval(model, line):
    def float_cvt(s):
        try:
            return float(s)
        except ValueError:
            return None

    prompt = build_mmoral_gpt4_prompt(line)
    log = ''
    retry = 5
    for i in range(retry):
        output = model.generate(prompt, temperature=i * 0.5)
        score = float_cvt(output)
        if score is None:
            log += f'Try {i}: output is {output}, failed to parse.\n'
        elif score < 0 or score > 1:
            log += f'Try {i}: output is {output}, invalid score: {score}.\n'
        else:
            log += 'Succeed'
            return dict(log=log, score=score)
    log += 'All 5 retries failed.\n'
    return dict(log=log, score=0.0)

def MMOral_acc(result_file):
    data = load(result_file)
    tot = defaultdict(lambda: 0)
    score = defaultdict(lambda: 0)
    lt = len(data)
    cate2_list = []
    for i in range(lt):
        item = data.iloc[i]
        cate = item['category']
        cate2 = cate.replace(',', '_')
        if cate2 not in cate2_list:
            cate2_list.append(cate2)
        grade = float(item['score'])
        cate_list = ['Teeth', 'Patho', 'HisT', 'Jaw', 'SumRec', 'Report']
        for capa in cate_list:
            if capa in cate:
                tot[capa] += 1
                score[capa] += grade
        tot['Overall'] += 1
        tot[cate2] += 1
        score['Overall'] += grade
        score[cate2] += grade

    res = defaultdict(list)
    res2 = defaultdict(list)
    cate_list.append('Overall')
    cate2_list.append('Overall')
    for k in cate_list:
        res['Category'].append(k)
        res['tot'].append(tot[k])
        res['acc'].append(score[k] / tot[k] * 100)
    for v in cate2_list:
        res2['Category'].append(v)
        res2['tot'].append(tot[v])
        res2['acc'].append(score[v] / tot[v] * 100)
    res = pd.DataFrame(res)
    res2 = pd.DataFrame(res2)
    return res, res2

def get_single_choice_prediction(response, all_choices, index2ans):
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    candidates = []

    for choice in all_choices:
        if f'({choice})' in response:
            candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:
            if f' {choice} ' in response:
                candidates.append(choice)
            elif f' {choice}.' in response:
                candidates.append(choice)
            elif f' {choice},' in response:
                candidates.append(choice)
    
    if len(candidates) == 0:
        for index, ans in index2ans.items():
            ans_str = str(ans)
            if ans_str in response:
                candidates.append(index)
    
    if len(candidates) > 0:
        positions = {}
        for c in candidates:
            pos = response.find(f' {c} ')
            if pos == -1:
                pos = response.find(f'({c})')
            if pos == -1:
                pos = response.find(str(index2ans[c]))
            if pos != -1:
                positions[c] = pos
        
        if positions:
            return min(positions.items(), key=lambda x: x[1])[0]
    
    return random.choice(all_choices)

# Base class with common functionality
class MMOralBase(ImageBaseDataset):
    
    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        tgt_path_z = []
        if isinstance(line['image'], list):
            for i in range(len(line['image'])):
                tgt_path = osp.join(self.img_root, f"{line['index']}--{i + 1}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'][i], tgt_path)
                tgt_path_z.append(tgt_path)
        else:
            tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
            if not read_ok(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path_z.append(tgt_path)
        return tgt_path_z

# Class for open-ended questions
class MMOral_Open(MMOralBase):
    TYPE = 'VQA'
    DATASET_URL = {
        'MMOral_Open': 'https://huggingface.co/datasets/EasonFan/MM-Oral/resolve/main/MM-Oral-VQA-Open-Ended.tsv'
    }
    
    DATASET_MD5 = {
        'MMOral_Open': 'd19de56024ae7ffe68fdf9a6dde9a602'
    }
    
    def build_prompt(self, line):
        tgt_path = self.dump_image(line)
        question = line['question']
        prompt = f'Question: {question}\nPlease provide a detailed and accurate answer to the question.'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs
    
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        model = judge_kwargs['model']
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        
        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=3, **judge_kwargs)
            assert model.working(), ('MMOral-Open-ended evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MMOral_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['score'] == v['score']
            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score, score_fine = MMOral_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        score_fine_pth = storage.replace('.xlsx', '_score_fine.csv')
        dump(score, score_pth)
        dump(score_fine, score_fine_pth)
        return score

# Class for closed-ended questions
class MMOral_Close(MMOralBase):
    TYPE = 'MCQ'
    
    DATASET_URL = {
        'MMOral_Close': 'https://huggingface.co/datasets/EasonFan/MM-Oral/resolve/main/MM-Oral-VQA-Closed-Ended.tsv'
    }
    
    DATASET_MD5 = {
        'MMOral_Close': '4f0ecacbebee564e6e2923d2d7acca7f'
    }
    
    def build_prompt(self, line):
        tgt_path = self.dump_image(line)
        question = line['question']
        
        options_prompt = 'Options:\n'
        for i in [['A', '1'], ['B', '2'], ['C', '3'], ['D', '4']]:
            option_value = str(line[f'option{i[1]}'])
            options_prompt += f"{i[0]}. {option_value}\n"
            
        prompt = (f'Question: {question}\n' + options_prompt
                + 'Please answer the above multiple-choice question by selecting the single correct option (A, B, C, or D). '
                + 'If the provided information is insufficient to determine a clear answer, please choose the most likely '
                + 'correct option based on the available data and your judgment.')

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs
    
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        detail_result_file = eval_file.replace(f'.{suffix}', '_detailed_acc.csv')

        if not osp.exists(result_file) or not osp.exists(detail_result_file):
            data = load(eval_file)
            assert 'answer' in data and 'prediction' in data
            data['prediction'] = [str(x) for x in data['prediction']]
            data['answer'] = [str(x) for x in data['answer']]

            tot = defaultdict(lambda: 0)
            score = defaultdict(lambda: 0)
            
            main_category_list = ['Teeth', 'Patho', 'HisT', 'Jaw', 'SumRec']
            categories = set()
            subcategories = set()
            
            for _, line in data.iterrows():
                category = line.get('category', 'unknown')
                categories.add(category)
                subcategory = category.replace(',', '_')
                subcategories.add(subcategory)
                
                for main_cat in main_category_list:
                    if main_cat in category:
                        tot[main_cat] += 1
                
                tot[category] += 1
                tot[subcategory] += 1
                tot['Overall'] += 1

            for i in tqdm(data.iterrows()):
                line = i[1]
                category = line.get('category', 'unknown')
                subcategory = category.replace(',', '_')
                
                index2ans = {
                    'A': line['option1'],
                    'B': line['option2'],
                    'C': line['option3'],
                    'D': line['option4']
                }
                
                fact_option = get_single_choice_prediction(line['prediction'], ['A', 'B', 'C', 'D'], index2ans)
                
                if fact_option == line['answer']:
                    for main_cat in main_category_list:
                        if main_cat in category:
                            score[main_cat] += 1
                    
                    score[category] += 1
                    score[subcategory] += 1
                    score['Overall'] += 1
            
            main_result = defaultdict(list)
            main_category_list.append('Overall')
            for cat in main_category_list:
                main_result['Category'].append(cat)
                main_result['tot'].append(tot[cat])
                main_result['acc'].append(score[cat] / tot[cat] * 100 if tot[cat] > 0 else 0)
            
            detailed_categories = list(categories) + ['Overall']
            detailed_result = defaultdict(list)
            for cat in detailed_categories:
                detailed_result['Category'].append(cat)
                detailed_result['tot'].append(tot[cat])
                detailed_result['acc'].append(score[cat] / tot[cat] * 100 if tot[cat] > 0 else 0)
            
            main_df = pd.DataFrame(main_result)
            detailed_df = pd.DataFrame(detailed_result)
            
            main_df = main_df.sort_values('Category')
            detailed_df = detailed_df.sort_values('Category')
            
            dump(main_df, result_file)
            dump(detailed_df, detail_result_file)

        result = pd.read_csv(result_file)
        return result

class MMOral_OMNI(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'MMOral_OMNI':
        'https://huggingface.co/datasets/OralGPT/MMOral-Omni-Bench/resolve/main/MMOral-Omni-Bench.tsv',
    }
    DATASET_MD5 = {
        'MMOral_OMNI': '0741b88554134a83fe7e40c8ac9cfa79',
    }

    def __init__(self, dataset='MMOral_OMNI', skip_noimg=False):
        if dataset != 'MMOral_OMNI':
            import warnings
            warnings.warn(
                'To evaluate on MMOral-OMNI, we would suggest `MMOral_OMNI` for the default setting.'
            )
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)


    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']

        ###### Handle No Image Case ######
        if line['image'] == 'nan':
            msgs = [dict(type='text', value=question)]
            return msgs
        ###### Add by bryce; END ######

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))

        return msgs
    
    # It returns a DataFrame
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mmoral import MMOral_Omni_auxeval, MMOral_Omni_acc
        
        model = judge_kwargs['model']
        storage = get_intermediate_file_path(eval_file, f'_{model}')
        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(max_tokens=16384, **judge_kwargs)
            # model = build_judge(max_tokens=3, **judge_kwargs) ; !!! invalid for [max_tokens=3] !!!
            assert model.working(), 'MMOral evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE
            
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MMOral_Omni_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['score'] == v[
                        'score']
            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score, score_fine = MMOral_Omni_acc(storage)
        score_pth = get_intermediate_file_path(storage, '_score', 'csv')
        score_fine_pth = get_intermediate_file_path(storage, '_score_fine', 'csv')
        dump(score, score_pth)
        dump(score_fine, score_fine_pth)
        return score