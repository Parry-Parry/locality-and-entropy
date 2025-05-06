from .run_beir import run_topics
import os
from fire import Fire


def batch_run_topics(
    file: str,
    model_directory: str,
    output_directory: str,
    batch_size: int = 512,
    dont_overwrite: bool = False,
):
    print(
        f"Running with models in {model_directory} and saving to {output_directory}"
    )
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    for model_name in os.listdir(model_directory):
        model_path = os.path.join(model_directory, model_name)
        run_topics(
            file,
            model_path,
            output_directory,
            batch_size=batch_size,
            dont_overwrite=dont_overwrite,
            cat="dot" not in model_name,
        )

    return "Done!"


if __name__ == "__main__":
    Fire(batch_run_topics)
