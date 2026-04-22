# Example ProxDash registry — reference

Pairs with `example_proxdash_model_configs.json` in this directory. Curated
config used by tests to exercise the ProxDash registry-loading path. Not the
bundled default (that's `v1.3.0.json`). Config origin is `PROXDASH`, so the
min-proxai-version gate runs.

## Models

Output columns: formats marked `(BE)` are `BEST_EFFORT`; everything else listed is `SUPPORTED`.

| Provider | Model | Sizes | Rec. | Input | Output | web_search | thinking |
|---|---|---|---|---|---|---|---|
| gemini | gemini-1.5-pro | [large] | — | text, image, audio, video | text, json, pydantic (BE) | — | — |
| gemini | gemini-2.5-flash | [medium] | ✓ | text, image, audio | text, json, pydantic | ✓ | ✓ |
| gemini | gemini-2.5-flash-lite | [small] | ✓ | text | text, json (BE) | — | — |
| gemini | gemini-2.5-pro | [large] | ✓ | text, image, document, audio, video | text, json, pydantic | ✓ | ✓ |
| gemini | gemini-3-pro | [large, largest] | ✓ | text, image, document, audio, video | text, json, pydantic, multi_modal | ✓ | ✓ |
| gemini | veo-3.0 | — | ✓ | text, image | video | — | — |
| openai | dall-e-3 | — | ✓ | text | image | — | — |
| openai | gpt-4-turbo | [medium] | — | text, image | text, json (BE) | — | — |
| openai | gpt-4o | [medium] | ✓ | text, image, document | text, json, pydantic | ✓ | — |
| openai | gpt-5 | [large] | ✓ | text, image, document | text, json, pydantic | ✓ | ✓ |
| openai | gpt-5-nano | [small] | ✓ | text | text, json (BE) | — | — |
| openai | o3 | [large, largest] | ✓ | text, image, document | text, json, pydantic | — | ✓ |
| openai | tts-1 | — | ✓ | text | audio | — | — |

## By size

- `small`: gpt-5-nano, gemini-2.5-flash-lite
- `medium`: gpt-4o, gemini-2.5-flash, gpt-4-turbo
- `large` (includes `largest`): gpt-5, gemini-2.5-pro, gemini-1.5-pro, o3, gemini-3-pro
- `largest`: o3, gemini-3-pro
- no size tags: dall-e-3, tts-1, veo-3.0

## By output format

- text: all 10 text models
- json: all 10 text models (BEST_EFFORT on gpt-5-nano, gemini-2.5-flash-lite, gpt-4-turbo)
- pydantic: 8 text models, SUPPORTED (7) + BEST_EFFORT (gemini-1.5-pro); not set on gpt-5-nano, gemini-2.5-flash-lite, gpt-4-turbo
- multi_modal: gemini-3-pro
- image: dall-e-3
- audio: tts-1
- video: veo-3.0

## By input format (beyond `text`)

- image: gpt-4o, gpt-5, o3, gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro, gpt-4-turbo, gemini-1.5-pro, veo-3.0
- document: gpt-4o, gpt-5, o3, gemini-2.5-pro, gemini-3-pro
- audio: gemini-2.5-flash, gemini-2.5-pro, gemini-3-pro, gemini-1.5-pro
- video: gemini-2.5-pro, gemini-3-pro, gemini-1.5-pro

## Special flags

- `is_recommended=false`: gpt-4-turbo, gemini-1.5-pro
- `output_format.json=BEST_EFFORT`: gpt-5-nano, gemini-2.5-flash-lite, gpt-4-turbo
- `output_format.pydantic=BEST_EFFORT`: gemini-1.5-pro
- `add_system_to_messages=true`: every gemini model
- no temperature/stop/n (reasoner): o3
- no parameters, messages, system_prompt: dall-e-3, tts-1, veo-3.0

## default_model_priority_list

1. gemini/gemini-3-pro
2. openai/o3
3. gemini/gemini-2.5-pro
4. openai/gpt-4o
5. gemini/gemini-2.5-flash

## Invariants

- 2 providers (openai, gemini), 13 models
- 2 non-recommended: gpt-4-turbo, gemini-1.5-pro
- 2 tagged `largest` (always paired with `large`): o3, gemini-3-pro
- 1 with `output_format.multi_modal`: gemini-3-pro
- 1 per media type: dall-e-3 (image), tts-1 (audio), veo-3.0 (video)
- 3 with `output_format.json=BEST_EFFORT`: gpt-5-nano, gemini-2.5-flash-lite, gpt-4-turbo
- 1 with `output_format.pydantic=BEST_EFFORT`: gemini-1.5-pro
- 5 entries in `default_model_priority_list`
