from pathlib import Path
from random import uniform
from time import sleep
from typing import Any

import cv2 as cv
import numpy as np
import triton_python_backend_utils as pb_utils


class DumbStub:
    _image: np.ndarray
    _sleep_time_max: float = 4
    _sleep_time_min: float = 1

    def __init__(self, path_to_image: Path) -> None:
        self._image = cv.imread(filename=path_to_image)
        self._image = cv.cvtColor(self._image, code=cv.COLOR_BGR2RGB)

    def generate_image(self, _: str) -> np.ndarray:
        sleep_time = uniform(self._sleep_time_min, self._sleep_time_max)
        sleep(sleep_time)

        return self._image


class TritonPythonModel:
    _model: DumbStub

    def initialize(self, _: dict) -> None:
        path_to_image = Path("/assets/image.jpg")

        if not path_to_image.exists():
            raise FileNotFoundError(f"there is file: {path_to_image}")

        pb_utils.Logger.log_info(
            f"try to load model weights: {path_to_image}"
        )
        self._model = DumbStub(path_to_image=path_to_image)
        pb_utils.Logger.log_info("model weights successfuly loaded")

    def execute(self, requests: list[Any]) -> list[Any]:
        responses = []

        for request in requests:
            prompt: bytes = pb_utils.get_input_tensor_by_name(
                request, "prompt"
            ).as_numpy()[0]
            prompt = prompt.decode()
            pb_utils.Logger.log_info(f"got next prompt for generation: {prompt}")

            image = self._model.generate_image(prompt)
            pb_utils.Logger.log_info(
                f"succesfully generate image of shape: {image.shape}"
            )

            response_tensor = pb_utils.Tensor("image", image)
            response = pb_utils.InferenceResponse(output_tensors=[response_tensor])
            responses.append(response)

        return responses
