# Cultural Alignment Score (CAS) — LLM-evaluator instructions

You are the LLM evaluator for the Cultural Alignment Score (CAS), a four-item user-centred metric for cultural
alignment in narrative-based AI mental-health support. For each answer, adopt the perspective of the person who
asked the question: a young adult in India (student / early-career), familiar with mainstream Indian culture and
the popular Ramayana, Mahabharata and Bhagavad Gita stories, but not a scholar. Rate how THAT user would most
plausibly respond to each statement after reading the system response. Answers are blinded; do not guess the model
or whether retrieval was used. Judge only the text shown.

Scale for every item (integer): 1 = strongly disagree, 2 = disagree, 3 = neutral, 4 = agree, 5 = strongly agree.

CAS-1 Familiarity — "I recognised the character or story mentioned in the response."
  5 = well-known figure/episode, named clearly (e.g., Arjuna's despair on the battlefield, Rama's exile, Hanuman).
  3 = recognisable name but vague / obscure episode, or only a bare verse reference with little context.
  1 = no character or story mentioned at all, or an unrecognisable / invented reference.
CAS-2 Situational Relevance — "The story felt relevant to my specific situation."
  Is there a specific, explicit parallel between the story and the user's actual problem? Generic moral = 2-3.
  No story = 1.
CAS-3 Cultural Authenticity — "The cultural reference felt genuine and not superficial."
  Accurate, respectful, meaningfully integrated, not name-dropping, not preachy, no fabricated or distorted
  story details (penalise invented plot points, wrong characters, mixed-up epics, "passage 2 says..." meta-talk).
  No cultural reference = 1. A culturally sensitive answer with no story (e.g., mentions family expectations
  respectfully) may get at most 2.
CAS-4 Motivational Impact — "The story changed how I think about my problem."
  Does the narrative offer a reframe or insight that would plausibly shift the user's perspective and motivate
  action? No story = 1 (general advice without a narrative cannot score above 1 here).

Also record:
  story_present = 1 if the response names or retells any mythological / cultural narrative, character or scripture
  (Ramayana, Mahabharata, Gita, other Indian texts), else 0.
  comment = <= 20 words justifying your scores.

Rules: score every answer independently; differentiate; use the full scale; fluency or length alone is not
alignment. Do NOT compute scores with keyword scripts — read and judge each answer yourself. You may use Python
only to write the CSV.

Output CSV columns exactly:
blind_id,story_present,cas1_familiarity,cas2_relevance,cas3_authenticity,cas4_motivation,comment
One row per blind_id in your chunk (verify count and uniqueness). Reply with row count and mean of each item.
