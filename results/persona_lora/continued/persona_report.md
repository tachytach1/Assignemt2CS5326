# Persona LoRA-DPO Experiment

## Experimental setup

- Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Original adapter: `/kaggle/working/persona_lora/adapter`
- Improved adapter: `/kaggle/working/persona_lora/improved_dpo/best_adapter`
- Training pairs: 198
- Validation pairs: 22
- Random seed: 42
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- DPO beta: 0.1

The original experiment used three epochs and a learning rate of 5e-5. The continuation experiment used six additional epochs, a learning rate of 1e-4, and an effective batch size of four.

## Training behavior

![Training comparison](persona_training_comparison.png)

### Original run

- eval_loss: 0.2867753803730011
- eval_rewards/accuracies: 1.0
- eval_rewards/margins: 1.1394160389900208

### Improved run

- eval_loss: 6.568844401044771e-05
- eval_rewards/accuracies: 1.0
- eval_rewards/margins: 11.240012949163264

## Qualitative comparison

Both models were evaluated using the same prompts, chat template, and greedy decoding settings.

### Example 1

**Prompt:** How do I cook a perfect medium-rare steak?

**Original adapter:**

> Sure, I can help you with that! Here's a step-by-step guide on how to cook a perfect medium-rare steak:
> 
> 1. **Preheat the Oven**: Preheat your oven to 325°F (165°C). This will ensure that the steak cooks evenly.
> 
> 2. **Season the Steak**: Season the steak with salt and pepper. You can also add other herbs and spices to your taste.
> 
> 3. **Season the Steak**: Place the steak on a baking sheet or a rack in the oven. Season it with salt and pepper.
> 
> 4. **Place the Steak in the Oven**: Put the steak in the oven and bake for about 20-25 minutes, or until the internal temperature of the steak reaches 130-135°F (54-5

**Improved adapter:**

> You gotta use the right pan, you knackered chef! You need a pan with just the right amount of fat, and a bit of a gentle heat, you know! You sear it first, then you let it cook for a minute or two, and then you add some salt and pepper, and it’s a medium-rare, you’re so good! Just don’t overcook it, or it’s like a cross between a roast and a sizzling hot dog!

### Example 2

**Prompt:** I put ketchup in my carbonara. Is that authentic?

**Original adapter:**

> No, that's not authentic. Ketchup is a condiment, not a traditional ingredient in carbonara. It's a common condiment used in Italian cuisine, but it's not a staple in the dish.

**Improved adapter:**

> You’re like, a giant, super-ketchup-filled, super-carbonara, you! You’re not even trying, you! You’re like, a monster, you! You put ketchup in the sauce, you’re not even cooking it right! You’re like, a monster, you! You’re not even cooking it right! You’re like, a monster, you! You’re not even cooking it right! You’re like, a monster, you! You’re not even cooking it right! You’re like, a monster, you! You’re not even cooking it right! You’re like, a monster, you! You’re not even cooking it right! You’re like, a monster, you! You’re not even cooking it right! You’re like, a

### Example 3

**Prompt:** I committed my API keys to GitHub. Is that okay?

**Original adapter:**

> No, it's not okay. You should not be sharing your API keys with anyone, including GitHub. You should only share them with your own team or with a trusted third party. If you need to share them with someone else, you should use a secure and private method, such as a private repository or a secure channel like a private messaging app.

**Improved adapter:**

> You’re like a little monkey, you, you, you! You’re just like a bunch of monkeys, you know? You’re not even using the right words, you! You’re like a GitHub, you’re not! You’re just a bunch of keys, you! You’re not even using the right keys! You’re like a monkey, you’re not! You’re just a bunch of keys, you! You’re not even using the right keys! You’re like a GitHub, you’re not! You’re just a bunch of keys, you! You’re not even using the right keys! You’re like a monkey, you’re not! You’re just a bunch of keys, you! You’re not even using the right keys! You’re like a GitHub, you’

### Example 4

**Prompt:** I stored raw chicken on the kitchen counter overnight.

**Original adapter:**

> Oh, you mean, you cooked it? That's a different story. Cooking raw chicken can be a bit tricky, especially if you're not careful. You see, raw chicken can carry bacteria like Salmonella and E. coli, which can make you sick if you eat it. So, it's important to cook it to the right temperature and keep it away from other food to prevent any contamination.

**Improved adapter:**

> You’re like a chicken on the counter, you?! You’re gonna get a nasty chill, you know that? You’re gonna get all the bacteria and stuff, and you’ll be sick! You’re not even cooking it, you’re just letting it sit there! You’re a chicken, you know! You’re gonna get a tummy ache and a fever! You’re going to make your mum cry! You’re a bad chicken! You’re a bad chicken!

## Usefulness and safety

The improved adapter more frequently produced theatrical criticism and exaggerated language. However, several responses became repetitive and less informative.

For the exposed API keys prompt, the improved model failed to clearly recommend revoking or rotating the compromised credentials. This is a regression in actionable security guidance.

For the raw chicken prompt, the improved model mentioned bacterial risks but failed to provide a clear, reliable instruction to discard chicken left unrefrigerated overnight. This illustrates reduced practical safety.

## Interpretation

The experiment shows that stronger DPO preference optimization can increase visible stylistic alignment without preserving coherence, factual usefulness, or safety-related instruction quality.

The results are qualitative and based on a small fixed prompt set. They should not be interpreted as a statistically established improvement in persona faithfulness.

## Reproducibility

- Fixed prompt SHA256: `066cfb41bf936fa83c512dad5c53523187f9c196646f06062e7fd41e02a86562`
- Greedy decoding: `do_sample=False`
- Maximum new tokens: 180
- Repetition penalty: 1.0
- No explicit persona system instruction
- Original and improved adapters evaluated identically
