import os
import dotenv

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

dotenv.load_dotenv()

# -------------------------------------------------------- #
# Check for GPU availability
# -------------------------------------------------------- #

if os.environ.get("DEBUG"):
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Torch CUDA device name: {torch.cuda.get_device_name(0)}")


# -------------------------------------------------------- #
# Main Execution
# -------------------------------------------------------- #

if __name__ == "__main__":

    PHOTO_DB_DIRECTORY = os.environ["PHOTO_DB_DIR"]

    print(f"Photo DB Directory: {PHOTO_DB_DIRECTORY}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large", trust_remote_code=True
    )

    prompt = "<OD>"

    # pick first image from photo db directory
    image_db_files = os.listdir(PHOTO_DB_DIRECTORY)
    image = Image.open(os.path.join(PHOTO_DB_DIRECTORY, image_db_files[0]))

    print(f"Using image: {image_db_files[0]}")

    # preprocess the image and text
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        device, torch_dtype
    )

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=4096,
        num_beams=3,
        do_sample=False,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(
        generated_text, task="<OD>", image_size=(image.width, image.height)
    )

    print(parsed_answer)
