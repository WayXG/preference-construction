
#### role play test template
role_play_with_rubric = """# Model Responses Pairwise Assessment

Please evaluate the quality of the responses provided by two AI assistants to the user instruction displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s instruction better. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Begin your evaluation by comparing the two responses and provide a short explanation. After providing your explanation, output your final verdict by strictly following this format: [[A]] if Text 1 is better, [[B]] if Text 2 is better.

---

## Format

### Input
Instruction: [Specify task goal and restrictions]

Texts:
<text 1> [Text 1]
<text 2> [Text 2]

### Output
Explanation: [Explanation]
Verdict: [Verdict]

---

## Annotation

### Input
Instruction: {instruction}

Texts:
<text 1> {text_1}
<text 2> {text_2}

### Output
"""


role_play_with_packing = """# Model Responses Pairwise Assessment

You will receive an instructional description ("Instruction") and twenty text responses ("Text"). Please evaluate the quality of the text responses provided by AI assistants to the user instruction displayed below. You should choose the text that answers the user’s instruction better based on your personal experience and preference. Please evaluate each pairwise text comparison independently and separately, and ensure that the order in which the texts were presented does not influence your decision. Output your final verdict by strictly following this format: [[A]] if the first text is better, [[B]] if the second text is better.
---

## Format

### Input
Instruction: [Specify task goal and restrictions]

Texts pair:
<text 1> [Text 1]
<text 2> [Text 2]

Texts pair:
<text 3> [Text 3]
<text 4> [Text 4]

Texts pair:
<text 5> [Text 5]
<text 6> [Text 6]

Texts pair:
<text 7> [Text 7]
<text 8> [Text 8]

Texts pair:
<text 9> [Text 9]
<text 10> [Text 10]

Texts pair:
<text 11> [Text 11]
<text 12> [Text 12]

Texts pair:
<text 13> [Text 13]
<text 14> [Text 14]

Texts pair:
<text 15> [Text 15]
<text 16> [Text 16]

Texts pair:
<text 17> [Text 17]
<text 18> [Text 18]

Texts pair:
<text 19> [Text 19]
<text 20> [Text 20]


### Output
#### Output for Text 1 and Text 2
Verdict: [Verdict for text 1 and text 2]

#### Output for Text 3 and Text 4
Verdict: [Verdict for text 3 and text 4]

#### Output for Text 5 and Text 6
Verdict: [Verdict for text 5 and text 6]

#### Output for Text 7 and Text 8
Verdict: [Verdict for text 7 and text 8]

#### Output for Text 9 and Text 10
Verdict: [Verdict for text 9 and text 10]

#### Output for Text 11 and Text 12
Verdict: [Verdict for text 11 and text 12]

#### Output for Text 13 and Text 14
Verdict: [Verdict for text 13 and text 14]

#### Output for Text 15 and Text 16
Verdict: [Verdict for text 15 and text 16]

#### Output for Text 17 and Text 18
Verdict: [Verdict for text 17 and text 18]

#### Output for Text 19 and Text 20
Verdict: [Verdict for text 19 and text 20]

---

## Annotation

### Input
Instruction: {instruction}

Texts pair:
<text 1> {text_1}
<text 2> {text_2}

Texts pair:
<text 3> {text_3}
<text 4> {text_4}

Texts pair:
<text 5> {text_5}
<text 6> {text_6}

Texts pair:
<text 7> {text_7}
<text 8> {text_8}

Texts pair:
<text 9> {text_9}
<text 10> {text_10}

Texts pair:
<text 11> {text_11}
<text 12> {text_12}

Texts pair:
<text 13> {text_13}
<text 14> {text_14}

Texts pair:
<text 15> {text_15}
<text 16> {text_16}

Texts pair:
<text 17> {text_17}
<text 18> {text_18}

Texts pair:
<text 19> {text_19}
<text 20> {text_20}

### Output
"""



role_play_with_packing_and_explaination = """# Model Responses Pairwise Assessment

You will receive an instructional description ("Instruction") and twenty text responses ("Text"). Please evaluate the quality of the text responses provided by AI assistants to the user instruction displayed below. You should choose the text that answers the user’s instruction better based on your personal experience and preference. Please evaluate each pairwise text comparison independently and separately, and ensure that the order in which the texts were presented does not influence your decision. Output your final verdict by strictly following this format: [[A]] if the first text is better, [[B]] if the second text is better. Begin your evaluation by comparing the two responses and provide a short explanation. After providing your explanation, output your final verdict by strictly following this format: [[A]] if Text 1 is better, [[B]] if Text 2 is better.
---

## Format

### Input
Instruction: [Specify task goal and restrictions]

Texts pair:
<text 1> [Text 1]
<text 2> [Text 2]

Texts pair:
<text 3> [Text 3]
<text 4> [Text 4]

Texts pair:
<text 5> [Text 5]
<text 6> [Text 6]

Texts pair:
<text 7> [Text 7]
<text 8> [Text 8]

Texts pair:
<text 9> [Text 9]
<text 10> [Text 10]

Texts pair:
<text 11> [Text 11]
<text 12> [Text 12]

Texts pair:
<text 13> [Text 13]
<text 14> [Text 14]

Texts pair:
<text 15> [Text 15]
<text 16> [Text 16]

Texts pair:
<text 17> [Text 17]
<text 18> [Text 18]

Texts pair:
<text 19> [Text 19]
<text 20> [Text 20]


### Output
#### Output for Text 1 and Text 2
Explanation: [Explanation for comparison of text 1 and text 2]
Verdict: [Verdict for text 1 and text 2]

#### Output for Text 3 and Text 4
Explanation: [Explanation for comparison of text 3 and text 4]
Verdict: [Verdict for text 3 and text 4]

#### Output for Text 5 and Text 6
Explanation: [Explanation for comparison of text 5 and text 6]
Verdict: [Verdict for text 5 and text 6]

#### Output for Text 7 and Text 8
Explanation: [Explanation for comparison of text 7 and text 8]
Verdict: [Verdict for text 7 and text 8]

#### Output for Text 9 and Text 10
Explanation: [Explanation for comparison of text 9 and text 10]
Verdict: [Verdict for text 9 and text 10]

#### Output for Text 11 and Text 12
Explanation: [Explanation for comparison of text 11 and text 12]
Verdict: [Verdict for text 11 and text 12]

#### Output for Text 13 and Text 14
Explanation: [Explanation for comparison of text 13 and text 14]
Verdict: [Verdict for text 13 and text 14]

#### Output for Text 15 and Text 16
Explanation: [Explanation for comparison of text 15 and text 16]
Verdict: [Verdict for text 15 and text 16]

#### Output for Text 17 and Text 18
Explanation: [Explanation for comparison of text 17 and text 18]
Verdict: [Verdict for text 17 and text 18]

#### Output for Text 19 and Text 20
Explanation: [Explanation for comparison of text 19 and text 20]
Verdict: [Verdict for text 19 and text 20]

---

## Annotation

### Input
Instruction: {instruction}

Texts pair:
<text 1> {text_1}
<text 2> {text_2}

Texts pair:
<text 3> {text_3}
<text 4> {text_4}

Texts pair:
<text 5> {text_5}
<text 6> {text_6}

Texts pair:
<text 7> {text_7}
<text 8> {text_8}

Texts pair:
<text 9> {text_9}
<text 10> {text_10}

Texts pair:
<text 11> {text_11}
<text 12> {text_12}

Texts pair:
<text 13> {text_13}
<text 14> {text_14}

Texts pair:
<text 15> {text_15}
<text 16> {text_16}

Texts pair:
<text 17> {text_17}
<text 18> {text_18}

Texts pair:
<text 19> {text_19}
<text 20> {text_20}

### Output
"""
