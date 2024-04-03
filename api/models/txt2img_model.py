from pydantic import BaseModel
from typing import Optional


# Definition of the Txt2ImgRequest model using Pydantic's BaseModel.
# This model specifies the parameters accepted by the txt2img function for generating images from text prompts.
# Details about what each parameter means are explained in the documentation.

class Txt2ImgRequest(BaseModel):
    api_key: Optional[str] = ""
    scheduler_name: Optional[str] = "unipc"         # dpmpp_sde_k, dpmpp_2m_k, unipc, ddim
    prompt: Optional[str] = "An image a happy couple walking along the beach, beautiful sunset, amazing full view, detailed, 8k"
    prompt2: Optional[str] = ""
    num_inference_steps: Optional[int] = 25
    seed: Optional[int] = -1
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    denoising_end: Optional[float] = 0.0
    guidance_scale: Optional[float] = 7.0
    negative_prompt: Optional[str] = "nude adult porn"
    negative_prompt_2: Optional[str] = "nude adult porn"
    num_images_per_prompt: Optional[int] = 1
    eta: Optional[float] = 0.0
    output_type: Optional[str] = "pil"
    return_dict: Optional[bool] = False
    guidance_rescale: Optional[float] = 0.0
    original_size: Optional[tuple[int, int]] = (1024, 1024)
    crops_coords_top_left: Optional[tuple[int, int]] = (0, 0)
    target_size: Optional[tuple[int, int]] = (1024, 1024)
    negative_original_size: Optional[tuple[int, int]] = (1024, 1024)
    negative_crops_coords_top_left: Optional[tuple[int, int]] = (0, 0)
    negative_target_size: Optional[tuple[int, int]] = (1024, 1024)
    model: Optional[str] = "base"   #"base", "refiner"
    refiner_prompt: Optional[str] = "high resolution, realistic, 8k, decent, beautiful, high quality"
