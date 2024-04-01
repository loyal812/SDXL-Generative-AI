import torch
from diffusers import DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, DDIMScheduler, UniPCMultistepScheduler


def load_scheduler(scheduler: str):
    main_config = {
        'beta_start': 0.00085,
        'beta_end': 0.012,
        'beta_schedule': 'scaled_linear',
        'use_karras_sigmas': True
    }
    match scheduler:
        case 'dpmpp_sde_k':
            # Recommend steps 10 ~ 15
            dpmpp_sde_k = DPMSolverSinglestepScheduler(**main_config)
            return dpmpp_sde_k
        case 'dpmpp_2m_k':
            # Recommend steps 20 ~ 30
            dpmpp_2m_k = DPMSolverMultistepScheduler(**main_config)
            model.scheduler = dpmpp_2m_k
            return "result2"
        case 'unipc':
            # Recommend steps 20 ~ 30
            unipc = UniPCMultistepScheduler(**main_config)
            return unipc
        case 'ddim':
            # Recommend steps 10 ~ 15
            # guidance_rescale=0.7
            ddim = DDIMScheduler.from_config(
                **main_config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
            )
            return ddim