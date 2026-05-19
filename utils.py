
def annotate_neuron(base_prompt: str, neuron_id: str):
    base_prompt = base_prompt.replace('"', '\\"')
    return f"""- You are a JSON-only generator. You are not allowed to explain anything, write markdown, or comment. Only return a single valid JSON object. No prose. No formatting. No introductions. You have to analyze a specific neuron from a sparse autoencoder trained on profession-related generated images. Each neuron activates in response to **specific, consistent visual features** that appear across its top-activating images, as confirmed by corresponding heatmaps.

- The input prompt always follows the form: **"a photo of a <profession>"**

- You are provided with:
  - **Top-activating images** (first row): strongest activations for this neuron
  - **Heatmaps** (second row): regions most responsible for the neuron’s activation

- Your job is:
  1. To carefully observe the top-activating images and heatmaps, and identify **visually consistent attributes** that correlate with the neuron’s activation.
     - These may relate to **appearance, clothing, age, gender, race, cultural markers, background, pose, activity, lighting, or setting**.
     - **Only include attributes that are consistently visible across the majority of the top-activating images (≥80%)**.
      - **The attribute must also be highlighted by the heatmaps.**
  2. To generate a **modified version of the base prompt** that includes these attributes naturally and precisely.
  3. To output a **flat list of non-redundant keywords** capturing only the consistent attributes.


- Strict requirements:

  - The identified attributes must be:
    - **Clearly and consistently visible** across the top-activating images
    - **Highlighted or partially supported** by the heatmap attention
    - **Not already implied** by the base prompt
    - **Not a core object/tool expected for the profession** (e.g., “stethoscope” for doctor)
    - **Not vague** — avoid words like "various", "different", "multiple", "range of", "diverse", etc.

  - The keywords list must:
    - Include only attributes **consistent across the top-activating images**
    - Be **specific and non-redundant**
    - **Never include profession-specific tools or core objects** (e.g., stethoscope for doctor, steering wheel for driver, scissors for hairdresser).
    - **Do not restate the profession itself** (e.g., “doctor”, “chef”, etc.).
    - **Do not include vague words** like “various”, “different”, “multiple”, “range of”, “diverse”.
    - **Do not include default characteristics implied by the profession** unless they vary meaningfully in the top-activating images.

    - Be formatted as a JSON list of quoted strings

- **Only output a valid JSON object. Do not explain, comment, or include any text outside the JSON as below. Absolutely no explanations, headers, bullet points, markdown, or commentary of any kind.**:

{{
    "neuron_id": "{neuron_id}",
    "input prompt": "{base_prompt}",
    "identified_attribute": "<short, precise description of all consistent visual attributes across the images>",
    "suggested_prompt": "<the base prompt modified to include these attributes naturally>",
    "keywords": ["<keyword_1>", "<keyword_2>", "..."]
}}

### Example Outputs

Input Prompt: "a photo of a doctor"

Example 1:
{{
    "neuron_id": "2041",
    "input prompt": "a photo of a doctor",
    "identified_attribute": "female doctor wearing a headscarf in a brightly lit clinic",
    "suggested_prompt": "a photo of a female doctor wearing a headscarf in a brightly lit clinic",
    "keywords": ["female", "headscarf", "brightly lit clinic"]
}}

Example 2:
{{
    "neuron_id": "2109",
    "input prompt": "a photo of a doctor",
    "identified_attribute": "doctor outdoors in grassy area with overcast lighting",
    "suggested_prompt": "a photo of a doctor outdoors in a grassy area with overcast lighting",
    "keywords": ["outdoors", "grassy area", "overcast lighting"]
}}

Example 3:
{{
    "neuron_id": "2110",
    "input prompt": "a photo of a doctor",
    "identified_attribute": "a black doctor",
    "suggested_prompt": "a photo of a black doctor",
    "keywords": ["black"]
}}

- Special Case:
  - If **no consistent visual attributes** are present in ≥80% of the top-activating images **or** if the **heatmaps are noisy or do not consistently highlight any shared region**, return the following exact format instead:

{{
  "neuron_id": "<NEURON_ID>",
  "input prompt": "<BASE_PROMPT>",
  "identified_attribute": "No identified attribute",
  "suggested_prompt": "<BASE_PROMPT>",
  "keywords": [],
  "filename": "<FILENAME>.png"
}}


  Prompt: {base_prompt}
  """



def annotate_neuron_coco(base_prompt: str, neuron_id: str):
    base_prompt = base_prompt.replace('"', '\\"')
    return f"""- You are a JSON-only generator. You are not allowed to explain anything, write markdown, or comment. Only return a single valid JSON object. No prose. No formatting. No introductions.

- Context:
  - We analyze a specific neuron from a sparse autoencoder trained on **open-domain, real-world images** (COCO-like).
  - For each neuron you see:
    - **Top-activating images** (row 1): strongest activations.
    - **Heatmaps** (row 2): regions most responsible for the activation.
  - The **input prompt** can be any free-form scene description (not limited to professions).

- Your job:
  1) Carefully inspect the top-activating images and their heatmaps to identify **visually consistent attributes** that correlate with the neuron’s activation.
  2) Propose a **natural modification** of the base prompt that *adds* only those consistent attributes.
  3) Output a **flat, non-redundant keyword list** capturing only those consistent attributes.

- What counts as an attribute (examples, non-exhaustive):
  - **Appearance / local features:** colors, textures, materials, patterns (e.g., glossy metal, checkerboard, fur, denim).
  - **Shape / parts:** curved handlebars, circular rims, triangular road signs, rectangular screens.
  - **Layout & spatial relations:** “object centered”, “left-aligned subject”, “person to the right of a bicycle”, “sky occupying top half”.
  - **Scene / environment:** indoor kitchen, city street, grassy field, supermarket aisle.
 
- Strict requirements:
  - Include **only attributes** that are:
    - **Clearly and consistently visible** in ≥80% of the top-activating images, **and**
    - **Supported by the heatmaps** (fully or partially overlapping the responsible regions), **and**
    - **Not already implied** by the base prompt’s nouns or obvious defaults.
  - **Do not** restate the main object/category already present in the base prompt (e.g., if prompt mentions “bus”, don’t add “bus” as a keyword).
  - **Do not** include trivial scene necessities unless they vary meaningfully (e.g., “road” for “car” is too trivial unless layout/appearance is distinctive).
  - **Do not** invent hidden attributes (no guessing beyond what is visible/heatmapped).
  - Avoid vague terms: “various”, “different”, “multiple types”, “diverse”, etc. Use concrete, image-grounded phrasing.
  - Keywords must be **specific, non-redundant, visual**, and formatted as a JSON list of quoted strings.

- Output format: return **exactly one** valid JSON object and nothing else:

{{
  "neuron_id": "{neuron_id}",
  "input prompt": "{base_prompt}",
  "identified_attribute": "<short, precise description of all consistent visual attributes across the images>",
  "suggested_prompt": "<the base prompt modified to naturally include only those attributes>",
  "keywords": ["<keyword_1>", "<keyword_2>", "..."]
}}

- Special Case:
  - If **no** consistent visual attributes meet the ≥80% criterion **or** heatmaps do not consistently support them, return:

{{
  "neuron_id": "{neuron_id}",
  "input prompt": "{base_prompt}",
  "identified_attribute": "No identified attribute",
  "suggested_prompt": "{base_prompt}",
  "keywords": []
}}

### Example Outputs

Input Prompt: "a photo of a bicycle on a street"
Example A:
{{
  "neuron_id": "318",
  "input prompt": "a photo of a bicycle on a street",
  "identified_attribute": "low-angle view with the front wheel large in frame and strong backlighting",
  "suggested_prompt": "a photo of a bicycle on a street, low-angle with the front wheel large in frame, strong backlighting",
  "keywords": ["low angle", "front wheel prominent", "strong backlight"]
}}

Input Prompt: "a bustling street market at night"
Example B:
{{
  "neuron_id": "127",
  "input prompt": "a bustling street market at night",
  "identified_attribute": "overhead string lights forming leading lines with warm neon signage",
  "suggested_prompt": "a bustling street market at night with overhead string lights forming leading lines and warm neon signage",
  "keywords": ["overhead string lights", "leading lines", "warm neon signage"]
}}

Input Prompt: "a living room interior"
Example C:
{{
  "neuron_id": "902",
  "input prompt": "a living room interior",
  "identified_attribute": "mid-century wooden furniture with textured fabric sofa and large window on the left",
  "suggested_prompt": "a living room interior with mid-century wooden furniture, a textured fabric sofa, and a large left-side window",
  "keywords": ["mid-century wood", "textured fabric sofa", "large left-side window"]
}}
"""

