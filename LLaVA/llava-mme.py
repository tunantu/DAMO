from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import os
model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device_map="cuda",
    device="cuda",
)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="baseline")
parser.add_argument("--tau", type=float, default="-0.3")
parser.add_argument("--beta_1", type=float, default="0.05")
parser.add_argument("--beta_2", type=float, default="0.20")
parser.add_argument("--alpha", type=float, default="0.7")

args1 = parser.parse_args()

output_dir = args1.output_dir


import os
import shutil
import subprocess

eval_tool_dir = "/path/to/MME_Benchmark_release_version/MME_Benchmark/eval_tool"
output_dir = os.path.join(eval_tool_dir, output_dir)
source_dir = os.path.join(eval_tool_dir, "Your_Results")  ## Here, Your_Results denotes the empty MME results dir.


if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  


shutil.copytree(source_dir, output_dir)




folders_and_files = [
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/artwork/images",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/artwork.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/celebrity/images",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/celebrity.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/code_reasoning",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/code_reasoning.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/color",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/color.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/commonsense_reasoning",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/commonsense_reasoning.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/count",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/count.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/existence",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/existence.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/landmark/images",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/landmark.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/numerical_calculation",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/numerical_calculation.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/OCR",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/OCR.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/position",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/position.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/posters/images",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/posters.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/scene/images",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/scene.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/MME_Benchmark/text_translation",
        "output_file_path": f"MME_Benchmark_release_version/MME_Benchmark/eval_tool/{output_dir}/text_translation.txt"
    },
]

for item in folders_and_files:
    image_folder = item["image_folder"]
    output_file_path = item["output_file_path"]

    with open(output_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        idx = 0
        for line in lines:
            print(f"Processing {idx} in {output_file_path}")
            idx += 1
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                image_file = parts[0]
                question = parts[1]
                image_path = os.path.join(image_folder, image_file)
                args = type('Args', (), {
                    "query": question,
                    "conv_mode": None,
                    "image_file": image_path,
                    "sep": ",",
                    "temperature": 0,
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 512,
                    "device_map":"cuda",
                    "device":"cuda",
                    "model": model,
                    "tokenizer": tokenizer,
                    "image_processor":image_processor,
                    "context_len":context_len,
                    "model_name": get_model_name_from_path(model_path),
                    "output_hidden_states": True,   ### This is necessary for DAMO.
                    "tau":args1.tau,
                    "beta_1":args1.beta_1,
                    "beta_2":args1.beta_2,
                    "alpha":args1.alpha,
                    
                })()
                output_text = eval_model(args)
                output_text = output_text.replace('\n', ' ')
                output_text = output_text.replace('\t', ' ')
                output_line = f"{line.strip()}\t{output_text}\n"
                output_file.write(output_line)

    print(f"Finished processing {output_file_path}")

print("All files have been processed!")



subprocess.run(["python", "calculation.py", "--results_dir", output_dir], cwd=eval_tool_dir)

