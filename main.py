import gradio as gr
from vllm import LLM, SamplingParams
prompt_template=f'''You are an casual AI bot for summarization, capable of rewriting certain inputs for clarity.You are a AI bot specialized in summarizing information concisely within two lines,give Only the summerized text ,don't show the input data,  You'll summarize the data within two lines.If the Days indicates 0 to 6 gives the result as day:0 for sunday,day:1 for monday ... upto day:6 for saturday and If the values exceeds 6 then starts from day:0.

If the Month indicates 0 to 11 gives the result as month:0 for january,monnth:1 for febrauary ... upto month:11 for december and If the values exceeds 11 then starts from month:0

### Instruction(Don't use any brackets and "In this given information"):
{{prompt}}

### Answer:
'''

llm = LLM(
    model="TheBloke/WestLake-7B-v2-AWQ",
    quantization="awq",
    max_model_len=2000,
    gpu_memory_utilization=0.9,
    dtype="half"
)
def generate_text(prompt):
    prompt_with_template = prompt_template.format(prompt=prompt)
    outputs = llm.generate([prompt_with_template], (SamplingParams(temperature=0.9,max_tokens=200)))
    generated_text = outputs[0].outputs[0].text if outputs else ""
    return generated_text
iface = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    title="AWQ Data Summarization",
    description="Enter a text to summarize.",
)
iface.launch()



"""
Earth is the third planet from the Sun and the only known celestial body with liquid surface water and life. It's a water world, with 70.8% of its surface being covered by oceans. Earth's crust consists of two main types: oceanic and continental, with the latter making up 29.2% of the planet's surface. Most of Earth's land is covered by vegetation, while large ice caps exist at the poles. Earth's atmosphere is primarily composed of nitrogen and oxygen, and its dynamic nature sustains surface conditions and protects the planet from most meteoroids and UV-light. Earth's crust is divided into tectonic plates, which interact to create mountains, volcanoes, and earthquakes. Earth's interior is divided into layers, with a solid crust, a highly viscous mantle, a transition zone, a liquid outer core, and a solid inner core.
"""