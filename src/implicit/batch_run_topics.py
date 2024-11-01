from .run_topics import run_topics
import os 
from fire import Fire

def batch_run_topics(ir_dataset : str, 
               model_directory : str, 
               output_directory : str,
               topics_or_res : str = None,
               batch_size : int = 512, 
               text_field : str = 'text', 
               overwrite : bool = False):
    print(f"Running {ir_dataset} with models in {model_directory} and saving to {output_directory}")
    if not os.path.exists(output_directory): os.makedirs(output_directory, exist_ok=True)
    formatted_dataset = ir_dataset.replace('-', '_').replace('/', '_')
    for model_name in os.listdir(model_directory):
        model_path = os.path.join(model_directory, model_name)
        out_path = os.path.join(output_directory, f"{formatted_dataset}_{model_name}.res.gz")
        run_topics(ir_dataset, model_path, out_path, topics_or_res=topics_or_res, batch_size=batch_size, text_field=text_field, overwrite=overwrite, cat='bi' not in model_name)

    return "Done!"

if __name__ == '__main__':
    Fire(batch_run_topics)