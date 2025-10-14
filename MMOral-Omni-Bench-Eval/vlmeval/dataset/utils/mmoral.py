from ...smp import *


def build_mmoral_gpt4_prompt(line):
    question = line['question']
    gt = str(line['answer'])
    prediction = str(line['prediction'])
    prompt = """
    You are an expert judge for a benchmark on oral and dental multimodal imaging diagnosis, examination, and treatment planning. Your task is to strictly evaluate how correct an AI model's prediction is compared to the ground truth. Just complete the last space of the correctness score without providing explanations or reasoning. The numeric correctness score must be one of 0.0 (totally wrong), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, or 1.0 (totally right).

    **Evaluation rules**
    - Only output the numeric score, nothing else. 
    - Treat clinically non-equivalent statements (e.g., wrong disease, wrong tooth or side, wrong stage or grade, wrong severity) as largely incorrect.
    - If the model’s prediction and the ground truth differ in their essential clinical content regarding the conditions or abnormalities, it should be treated as a major error.
    - If the ground truth includes fine-grained coordinates or tooth numbers that are omitted in the model’s prediction, but their clinical meaning remains consistent, treat it as a minor omission.
    - Judge semantic equivalence, not wording. Accept clinically equivalent synonyms (e.g., “caries” ≈ “tooth decay”; “periapical radiolucency” ≈ “apical rarefaction”).
    - Accept equivalent tooth numbering systems (FDI, Universal, Palmer).
    - Do not over-penalize minor phrasing differences; focus on clinical meaning.
    - If the model’s prediction includes additional statements that contradict the GT or introduce clinically false information, treat it as a major error, even if the rest matches.
    - The quality of the model’s prediction is determined by how closely it aligns with the ground truth, not by its length or verbosity. Both insufficient and excessive content should result in score penalties.

    **Scoring guide**  
    * 1.0: Fully correct; matches GT in all clinically material aspects; no extra wrong claims.
    * 0.7–0.9: Mostly correct; only minor omissions or clinically insignificant inaccuracies; no contradiction to GT.
    * 0.3–0.6: Partially correct; some correct elements but misses at least one core aspect or introduces a major error.
    * 0.1–0.2: Largely incorrect; minimal overlap with GT.
    * 0.0: Totally incorrect; no meaningful overlap; refusal to answer; generic non-answer; or answer contradicts the GT fundamentally.

    **Example format:**

    Question | Ground truth | Prediction | Correctness
    --- | --- | --- | ---
    This is an intraoral photograph of the oral cavity. What is the likely diagnosis? | Findings are most consistent with a minor aphthous ulcer (minor aphthous stomatitis) on the ventral anterior tongue. | The image shows a small, shallow sore on the underside of the front part of the tongue, suggestive of a minor aphthous ulcer. | 1.0
    This is an intraoral photograph. What disease(s) is demonstrated here? | The permanent tooth decay is detected in regions <box>[81,90,353,339]</box> and <box>[108,267,401,497]</box>. | This intraoral image shows the presence of dental caries (tooth decay) on multiple molars, characterized by brownish and darker areas on the occlusal surfaces of the teeth. The decay appears to be advancing, indicating active carious lesions. | 0.8
    The pathological image from the gingiva demonstrates a lesion measuring 4.5 cm. Evaluate this pathological patch's morphology and indicate if it corresponds to leukoplakia without dysplasia, leukoplakia with dysplasia, or oral squamous cell carcinoma. | Histopathological findings indicate leukoplakia without dysplasia. | The lesion most closely corresponds to leukoplakia with dysplasia or possibly early oral squamous cell carcinoma** depending on invasion, but based purely on this image, leukoplakia with dysplasia is a reasonable assessment. | 0.5
    This is an intraoral photograph of the oral cavity. What condition is visible? | Mixed dentition period, the upper central incisor space and mandibular space can be seen. | The image shows a case of severe dental crowding and malalignment of the teeth, likely due to malocclusion. The teeth are misaligned, overlapping, and displaced, which appears to be a case of advanced dental crowding. This condition often results from insufficient space in the dental arch for the erupted teeth. | 0.2
    This is a periapical X-ray image. Please identify the disease(s) or condition(s). | Periapical radiolucency at the apex of the posterior tooth consistent with chronic apical periodontitis (periapical granuloma versus small radicular cyst) associated with a non‑endodontically treated tooth; correlate clinically and consider endodontic therapy. | This periapical X-ray image shows multiple affected teeth with signs consistent with dental caries (cavities). There are no obvious signs of advanced periodontal disease or other abnormalities. | 0.0
    """

    gpt4_prompt = prompt + '\n' + ' | '.join(
        [question, gt.replace('<AND>', ' <AND> ').replace('<OR>', ' <OR> '), prediction, ''])
    return gpt4_prompt

def extract_box_content(text):
    # 使用非贪婪模式匹配 <|begin_of_box|> 与 <|end_of_box|> 之间的内容
    pattern = r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>"
    matches = re.findall(pattern, text, re.DOTALL)  # DOTALL 允许匹配跨多行内容
    return [m.strip() for m in matches][0]

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
        # cate_list = ['teeth', 'patho', 'his', 'jaw', 'summ', 'report']
        # cate_list = ['rec', 'ocr', 'know', 'gen', 'spat', 'math', 'teeth', 'patho', 'his', 'jaw', 'summ', 'report']
        cate_list = ['TE', 'Oral Histopathology', 'Oral Mucosal Disease', 'Oral & Maxillofacial Radiology', 
                     'TP', 'Endodontics', 'Implant Dentistry', 'Periodontics',
                     'II_Loc',
                     'II_Dx-I', 'Orthodontics', 'Cancer', 'Gingivitis', 'Defective Dentition', 'Normality', 'Tooth Discoloration', 'Ulcer', 'Caries', 'Calculus',
                     'II_Dx-R', 'Fenestration and Dehiscence', 'Malocclusion Issues Assessment', # "Caries", 
                     'PA', 'Impacted Tooth', 'Pulpitis', 'Periodontitis', 'Apical Periodontitis', 'Mixed Dentition', 'Bone Loss', 'Root Canal Treatment', 'Crown', 'Restoration', 
                     'CE',
                     'PI', 'Leukoplakia with Dysplasia', 'Leukoplakia without Dysplasia', 'Oral Squamous Cell Carcinoma', 'Oral Submucous Fibrosis', # "Normal", 
                     'IV'
                     ]

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
        res['acc'].append(score[k] / tot[k] * 100 if tot[k] else 0)
    for v in cate2_list:
        res2['Category'].append(v)
        res2['tot'].append(tot[v])
        res2['acc'].append(score[v] / tot[v] * 100 if tot[k] else 0)
    res = pd.DataFrame(res)
    res2 = pd.DataFrame(res2)
    return res, res2
