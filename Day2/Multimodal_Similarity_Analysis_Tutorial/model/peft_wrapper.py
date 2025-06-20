from typing import List


def get_lora_config(
        lora_r: int = 8,
        lora_alpha: int = 1,
        lora_dropout: float = 0.1,
        lora_bias: str = 'none', 
        lora_target_modules: List[str] = ["q_proj", "v_proj"],
    ):
    """ Setup LORA config using PEFT library. 
    """
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        target_modules=list(lora_target_modules),
    )
    return peft_config


def setup_peft_model(model, peft_config: PeftConfig):
    """ Creates pytorch model with peft adapter layers injested.
        The original model is also overwritten with PEFT adapters.

    Args:
        model: pytorch model
        peft_config: peft configuration instance
    """
    for param in model.parameters():
        param.requires_grad = False

    model = inject_adapter_in_model(peft_config, model)

    return model