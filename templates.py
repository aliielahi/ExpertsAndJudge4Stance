from jinja2 import Template

TEMPLS = {
"QU" : """Detect the stance of the writer about $topic$ and \
respond with 'Neutral', 'Support', or 'Against'\n#Answer:""",
"INST" : """In the following sentence, is the stance of writer \
against $topic$ or supports it or neutral to it?\n""",
## trying
"QU2" : """\nDetect the stance of the writer about $topic$ and give a single word \
respond with 'against', or 'support'.\n#Answer:""",
"INST2" : """In the following sentence, is the point of view of the writer \
against the $topic$ or supports it?\n"""
}

expert_support = Template("""Consider the following sentence about {{ title }}:
{{ sentence }}

We identified the writer's stance as supportive of {{ title }}. In one sentence, explain and debate about why the writer's stance is supportive of {{ title }}.\n##Answer:""")

expert_against = Template("""Consider the following sentence about {{ title }}:
{{ sentence }}

We identified the writer's stance against {{ title }}. In one sentence, explain and debate about why the writer's stance is against {{ title }}.\n##Answer:""")

expert_neutral = Template("""Consider the following sentence about {{ title }}:
{{ sentence }}

We identified the writer's stance neutral regarding {{ title }}. In one sentence, explain and debate about why the writer's stance is neutral regarding {{ title }}.\n##Answer:""")

to_judge_with_sentence_old = Template("""We consulted three experts to determine the stance of the following sentence about {{ title }}:
{{ sentence }}

Each expert was tasked with supporting one of the following classes: supportive, neutral, or against.
Please analyze the experts' opinions for each class and determine the overall stance of the sentence. The experts provided the following explanations:

**Experts' perspectives:**
- **Supports {{ title }}:** {{ suppp }}
- **Against {{ title }}:** {{ agaaa }}
- **Neutral about {{ title }}:** {{ neuuu }}

Based on the experts' opinions and the given sentence, determine the author's stance regarding {{ title }}. Give a single word response with "Against", "Support", or "Neutral".\n##Answer:""")

to_judge_with_sentence = Template("""We consulted three experts to determine the stance of the following sentence about {{ title }}:
{{ sentence }}

Each expert was tasked with supporting one of the following classes: supportive, neutral, or against.
Please analyze the experts' opinions for each class and determine the overall stance of the sentence. The experts provided the following explanations:

**Experts' perspectives:**
- **Supports {{ title }}:** {{ suppp }}
- **Against {{ title }}:** {{ agaaa }}
- **Neutral about {{ title }}:** {{ neuuu }}

First Try to detect the stance of the sentence, then use the experts' opinions to make sure about your decision. Give only one word as response with one of "Against", "Support", or "Neutral".\n##Answer:""")


to_judge_no_sentence = Template("""We consulted three experts to determine the stance of a sentence about {{ title }}:
Each expert was tasked with supporting one of the following classes: supportive, neutral, or against.
Please analyze the experts' opinions for each class and determine the overall stance of the sentence. The experts provided the following explanations:

**Experts' perspectives:**
- **Supports {{ title }}:** {{ suppp }}
- **Against {{ title }}:** {{ agaaa }}
- **Neutral about {{ title }}:** {{ neuuu }}

Based on the experts' opinions and the given sentence, determine the author's stance regarding {{ title }}. Give a single word response with "Against", "Support", or "Neutral".\n##Answer:""")