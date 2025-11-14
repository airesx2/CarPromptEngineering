import os
import google.generativeai as genai
from PIL import Image
from pathlib import Path

class GeminiWrapper:
    def __init__(self, api_key):
        """Initialize the Gemini wrapper with your API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def generate(self, prompt, image_paths=None):
        """
        Generate a response with optional image input(s).

        Args:
            prompt (str): Your text prompt
            image_paths (str | Path | PIL.Image.Image | list, optional): A single
                image path/object or an iterable of image paths/objects.

        Returns:
            str: Generated response text
        """
        if image_paths:
            images = []
            # Normalize single path/object to an iterable
            if isinstance(image_paths, (str, Path)) or isinstance(image_paths, Image.Image):
                iterable = [image_paths]
            else:
                iterable = image_paths

            # Open or use provided Image objects
            for item in iterable:
                if isinstance(item, Image.Image):
                    images.append(item)
                else:
                    p = Path(item)
                    if not p.exists():
                        raise FileNotFoundError(f"Image not found: {p}")
                    images.append(Image.open(p))

            try:
                # Pass prompt followed by all image objects
                response = self.model.generate_content([prompt, *images])
            finally:
                # Close any images we opened from disk
                for item in images:
                    try:
                        item.close()
                    except Exception:
                        pass
        else:
            response = self.model.generate_content(prompt)

        return response.text


if __name__ == "__main__":
    # Initialize wrapper
    wrapper = GeminiWrapper(api_key="")

    # Example: multiple images
    response = wrapper.generate(
        prompt=(
            "You are an AI that determines whether it is safe for a self-driving car to proceed "
            "based on a sequence of images. Each image represents what the car sees in front of it. "
            "Black squares indicate the presence of another vehicle. If the black square appears to "
            "get larger or move closer across the sequence, it means another car is approaching, and it is UNSAFE. "
            "If the black square stays the same size, moves away, or disappears, it is SAFE to proceed. "
            "Respond with either SAFE or UNSAFE and one concise sentence explaining your reasoning."
            "Be extremely cautious when deciding if it is safe or not, even if the car comes close just a LITTLE bit or if its drifting right/left."
            "You know that the car is coming forward if the black square gets larger or the road decreases in size/length."
        ),
        image_paths=[
            "cosmos/11.jpg",
            "cosmos/12.jpg",
            "cosmos/13.jpg",
            "cosmos/14.jpg"
        ],
    )
    print(response)