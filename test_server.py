import matplotlib.pyplot as plt
import numpy as np
import tritonclient.http as triton_http


def get_image(
    triton_client: triton_http.InferenceServerClient,
    model_name: str,
    model_version: str,
) -> np.ndarray | None:
    if not triton_client.is_model_ready(
        model_name=model_name,
        model_version=model_version,
    ):
        return None
        
    prompt = np.array(["beautiful picture"], dtype=np.object_)
    prompt_tensor = triton_http.InferInput(
        name="prompt",
        shape=list(prompt.shape),
        datatype=triton_http.np_to_triton_dtype(prompt.dtype),
    )
    prompt_tensor.set_data_from_numpy(prompt)

    response = triton_client.infer(
        model_name=model_name,
        model_version=model_version,
        inputs=[prompt_tensor],
    )
    return response.as_numpy(name="image")


def main() -> None:
    triton_client = triton_http.InferenceServerClient(
        url="127.0.0.1:8000"
    )
    model_name = "dumb_stub"
    model_version = "1"
    print("Send test request")
    print("Wait for response...")

    if (
        image := get_image(triton_client, model_name, model_version)
    ) is None:
        print("Model is not ready... Check your server")
        return
    
    print("Got image from server")
    plt.imshow(image)
    plt.show()

    
if __name__ == "__main__":
    main()
