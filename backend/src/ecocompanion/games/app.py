"""Standalone local entrypoint reusing the pet service and its security boundary."""

from ecocompanion.pet.api import create_pet_app

app = create_pet_app()
