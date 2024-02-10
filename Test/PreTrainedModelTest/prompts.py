def zephyr_p1(review):
    prompt = f"""<|user|>
Classify the following review as 'J', 'W', or 'V', where:
'J' is positive or encouraging or constructive or specific or has a positive tone or consists of factual statements and a positive tone.
'J' is positive.
'J' is encouraging.
'J' is constructive.
'J' is specific.
'J' has a positive tone.
'J' consists of factual statements and a positive tone.
'V' contains personal attacks
'V' contains excessive negativity without constructive feedback.
'W' is not encouraging.
'W' is not discouraging.
'W' consists of factual statements.

Review: "{review}"

Please classify this review as either 'J', 'W', or 'V'. Only output 'J', 'W', or 'V' with no additional explanation. Your classification:
<|endoftext|>
<assistant>
"""
    return prompt

def mistral_p1(review):
    prompt = f"""<s>[INST]
Classify the following review as 'J', 'W', or 'V', where:
'J' is positive or encouraging or constructive or specific or has a positive tone or consists of factual statements and a positive tone.
'V' contains personal attacks and a discouraging tone or excessive negativity without constructive feedback.
'W' meets the standards of professionalism and is not discouraging and is not encouraging or consists of factual statements.

Review: "{review}"

Please classify this review as either 'J', 'W', or 'V'. Only output 'J', 'W', or 'V' with no additional explanation. [/INST] Your classification: """
    return prompt