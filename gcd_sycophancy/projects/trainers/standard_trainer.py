from .base_trainer import BaseTrainer


class StandardTrainer(BaseTrainer):
    """
    Standard SFT trainer.

    Dataset preparation, optimization, checkpointing, and the epoch loop now
    live in BaseTrainer. This subclass exists for the default loss/step logic
    and for future overrides that need custom gradient computation.
    """

    pass
