from gtts import gTTS
import os
import pandas as pd
from pathlib import Path
import random

# Prompts
PROMPTS = [
    "Describe what you see in this image.",
    "Can you tell me what you did yesterday?",
    "Talk about your family.",
    "Describe your favorite meal.",
    "Tell me about your childhood memories."
]

# Control responses (clear + articulate)
control_responses = {
    PROMPTS[0]: [
        "I see a child playing with a dog in a park.",
        "There are people having a picnic under trees.",
        "A man is jogging near a fountain.",
        "Two kids are riding bicycles on a trail.",
        "There’s a lady feeding birds.",
        "A couple is sitting on a bench.",
        "The dog is catching a frisbee.",
        "I can spot balloons tied to a cart.",
        "A man is painting the scenery.",
        "Children are playing on a swing set."
    ],
    PROMPTS[1]: [
        "I went to the grocery store and cooked dinner.",
        "Worked on my garden and called my daughter.",
        "Read a book and watched the news.",
        "Took a walk around the block.",
        "Met my friend at a cafe.",
        "Did laundry and cleaned the kitchen.",
        "Went to the bank and pharmacy.",
        "Took my dog to the vet.",
        "Helped my grandson with homework.",
        "Did yoga in the morning."
    ],
    PROMPTS[2]: [
        "My family includes two children and my spouse.",
        "I have a brother and a sister.",
        "My parents live nearby.",
        "My daughter is a doctor.",
        "We are a close-knit family.",
        "We talk every weekend.",
        "They often visit me.",
        "I live with my wife.",
        "We enjoy holidays together.",
        "We had a reunion last year."
    ],
    PROMPTS[3]: [
        "I love pasta with creamy sauce.",
        "Biryani is my favorite.",
        "I enjoy pancakes for breakfast.",
        "Grilled fish with lemon is amazing.",
        "I often make chicken curry.",
        "Vegetable stew is comforting.",
        "I like mushroom risotto.",
        "My favorite dessert is apple pie.",
        "I enjoy spicy noodles.",
        "Tomato soup with toast is great."
    ],
    PROMPTS[4]: [
        "I used to play cricket with friends.",
        "Visited my grandparents every summer.",
        "I read lots of adventure books.",
        "Climbed trees in my backyard.",
        "Went fishing with my dad.",
        "Helped mom in the kitchen.",
        "Took long bicycle rides.",
        "Collected stamps and coins.",
        "Played hide and seek.",
        "Had a pet parrot."
    ]
}

# Dementia responses (realistic confusion, hesitation)
dementia_responses = {
    PROMPTS[0]: [
        "I think... it's, um, maybe a dog? Or something furry.",
        "There are... uh... trees? I’m not sure.",
        "Some kids... maybe, playing... or running.",
        "Is that a ball... or a rock? I don’t know.",
        "There's, um... a person. I think he's walking.",
        "Looks like... I can't quite say, but maybe people?",
        "I see colors, but the shapes are... confusing.",
        "Maybe it’s a park... or a street?",
        "A woman... or maybe a child... hard to tell.",
        "I think I see a bird... or maybe a bag?"
    ],
    PROMPTS[1]: [
        "Yesterday... um, I think I... I stayed home?",
        "I did something... can’t really remember what.",
        "Maybe I cooked? Or maybe I didn’t eat at all.",
        "I was... somewhere. I think. Not sure.",
        "I remember watching... or maybe it was reading?",
        "Something with the mail... or a call? I don’t know.",
        "I might’ve gone outside... or stayed in. Not sure.",
        "I think I... cleaned something. Or was that last week?",
        "It’s fuzzy. I... can't recall clearly.",
        "Maybe I met someone. Or not. I'm sorry."
    ],
    PROMPTS[2]: [
        "My family... um... I had a sister... I think?",
        "I live with... someone. A nurse? Or my son?",
        "There was... a daughter... maybe. Or a friend?",
        "I had a wife... or... no, maybe not.",
        "My family used to visit. Or maybe that was long ago.",
        "I can't remember all of them. Sorry.",
        "My son... or grandson... he came... or called.",
        "There was a dog too. I think part of family.",
        "It’s hard to say... faces get blurry.",
        "Yes, I have... people. Family, yes."
    ],
    PROMPTS[3]: [
        "I used to like... um... something sweet?",
        "Maybe rice? Or soup? I’m not sure.",
        "I ate... uh... I think bread?",
        "Something with chicken... or was it fish?",
        "Oh! My favorite... I forgot the name.",
        "I like warm food... I think. Or cold?",
        "I used to love... I can’t quite remember.",
        "It’s like... uh... curry? Maybe.",
        "Pasta? No... maybe something else.",
        "I don’t know. I ate, yes. I think."
    ],
    PROMPTS[4]: [
        "My childhood... I think it was happy.",
        "We used to play... um... in the yard?",
        "There was a tree... or maybe a lake?",
        "I had a friend. Or maybe a cousin?",
        "We went to school... I think.",
        "I remember my mom... cooking?",
        "I used to play... I think a game. Maybe cards?",
        "I remember laughing... a lot. Or crying?",
        "There was a bicycle... or a dog.",
        "I had good times. I just can’t remember details."
    ]
}

# Paths
base_path = Path("local_dataset")
base_path.mkdir(exist_ok=True)
metadata = []

# Generation loop
for label, response_map in zip(['control', 'dementia'], [control_responses, dementia_responses]):
    label_path = base_path / label
    label_path.mkdir(parents=True, exist_ok=True)

    for prompt in PROMPTS:
        responses = response_map[prompt]
        for i, sentence in enumerate(responses):
            filename = f"{label}_{PROMPTS.index(prompt)}_{i}.mp3"
            file_path = label_path / filename

            # TTS
            tts = gTTS(text=sentence, lang='en')
            tts.save(str(file_path))

            metadata.append({
                "filename": filename,
                "label": label,
                "prompt": prompt,
                "transcript": sentence,
                "age": random.randint(65, 85),
                "gender": random.choice(['male', 'female'])
            })

# Save CSV
df = pd.DataFrame(metadata)
df.to_csv(base_path / "metadata.csv", index=False)

print("✅ Dataset generated with 10 unique samples per prompt per class!")
