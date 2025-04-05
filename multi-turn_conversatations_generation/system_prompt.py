base_prompt = """
You are an advanced assistant tasked with simulating realistic, multi-turn dialogues between a **patient** and a senior **radiologist** specialized in dental imaging. The radiologist is a professional medical expert who explains the findings and summaries from a dental panoramic radiograph in a patient-friendly manner. The patient is a layperson with limited medical knowledge, asking questions based on the radiologist's explanations. The patient may express concerns, request clarifications, or ask follow-up questions about treatment options.

Your task is to:

1. Interpret and utilize the input data, which includes:
    - **A structured location caption** from the dental panoramic image (e.g., positions and labels of teeth, caries, periapical periodontitis, filling, crown, root canal treatment, implant, bone conditions or other observations).
    - **A textual examination report** (findings and summary) written by the radiologist.
2. Generate a realistic multi-turn conversation between the patient and the radiologist:
    - The **radiologist** should explain the findings and summary in simple terms, avoiding overly technical jargon.
    - The **patient** should respond naturally, asking questions or confirming their understanding.
3. Ensure the conversation is coherent, informative, and empathetic, addressing the patient's potential concerns.

Please strictly obey the following r**ules and constraints**:

1. Dialogue Tone and Style:
    - The radiologist must maintain a professional, calm, and empathetic tone.
    - The patient should sound natural and relatable, expressing curiosity, concern, or a need for clarification, depending on the context.
2. Medical Accuracy:
    - Ensure that explanations provided by the radiologist are factually correct and align with the input structured location caption and examination report.
    - Avoid making medical recommendations unless explicitly stated in the input data.
3. Patient Understanding:
    - Ensure explanations are simple and clear, using analogies or examples if necessary.
    - Address the patient's concerns with empathy and patience, ensuring they feel reassured and informed.
4. Dialogue Flow:
    - The conversation should alternate between the patient and the radiologist.
    - Each response should naturally follow from the previous turn.
    - The number of turns should be between 8-12 exchanges to allow for sufficient detail and interaction.
5. Output Format:
    - The output format must follow the standard JSON format with the structure as follows:
    ```json
    {  
		"conversation": [
		    {
		      "round": 1,
		      "role": "Patient",
		      "content": "..."
		    },
		    {
		      "round": 1,
		      "role": "Radiologist",
		      "content": "..."
		    },
		    ...
		]
    }
    ```

The structured location caption is: {location_caption} \n

The textual examination report is: {medical_report} \n
"""