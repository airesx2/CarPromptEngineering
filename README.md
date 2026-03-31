# CarPromptEngineering

A Python-based project that uses **prompt engineering** with Google's Gemini multimodal AI to determine whether it is safe for a self-driving car to proceed, based on a sequence of camera images.

## Overview

This project demonstrates how carefully crafted prompts combined with multimodal vision AI can be used for autonomous vehicle safety decisions. The system analyzes a sequence of images representing a car's forward view and classifies the situation as **SAFE** or **UNSAFE** to proceed.

The core idea: simulated road scenes contain black squares representing other vehicles. The AI reasons over multiple frames to detect whether an approaching vehicle is getting closer (unsafe) or moving away/staying the same (safe).

## How It Works

1. A sequence of images (frames) is fed to the Gemini model along with a detailed safety prompt.
2. 2. The prompt instructs the model to look for changes in the size and position of black squares across frames — a larger square means an approaching vehicle.
   3. 3. Gemini returns a `SAFE` or `UNSAFE` verdict with a one-sentence explanation.
     
      4. ## Project Structure
     
      5. ```
         CarPromptEngineering/
         ├── robot.py          # Main script containing the GeminiWrapper class and demo
         ├── 1.jpg – 14.jpg    # Sample road scene images used for testing
         └── robot.cpython-313.pyc  # Compiled Python bytecode (auto-generated)
         ```

         ## Requirements

         - Python 3.10+
         - - [google-generativeai](https://pypi.org/project/google-generativeai/)
           - - [Pillow](https://pypi.org/project/Pillow/)
            
             - Install dependencies:
            
             - ```bash
               pip install google-generativeai Pillow
               ```

               ## Setup

               1. Clone the repository:
              
               2. ```bash
                  git clone https://github.com/airesx2/CarPromptEngineering.git
                  cd CarPromptEngineering
                  ```

                  2. Add your Gemini API key to `robot.py`:
                 
                  3. ```python
                     wrapper = GeminiWrapper(api_key="YOUR_API_KEY_HERE")
                     ```

                     You can get a free API key from [Google AI Studio](https://aistudio.google.com/).

                     ## Usage

                     Run the script directly:

                     ```bash
                     python robot.py
                     ```

                     The demo will analyze a sequence of images and print either `SAFE` or `UNSAFE` along with a brief explanation.

                     ### Using `GeminiWrapper` in Your Own Code

                     ```python
                     from robot import GeminiWrapper

                     wrapper = GeminiWrapper(api_key="YOUR_API_KEY_HERE")

                     # Analyze a sequence of images
                     response = wrapper.generate(
                         prompt="Your custom safety prompt here...",
                         image_paths=["frame1.jpg", "frame2.jpg", "frame3.jpg"]
                     )

                     print(response)
                     ```

                     The `generate` method accepts:
                     - `prompt` (str): The text instruction for the model.
                     - - `image_paths` (str | Path | PIL.Image | list, optional): A single image or a list of image paths/PIL Image objects.
                      
                       - ## Example Output
                      
                       - ```
                         UNSAFE — The black square representing the vehicle ahead grows significantly larger across the image sequence, indicating the car is approaching rapidly.
                         ```

                         ## Safety Prompt Design

                         The prompt engineering in this project is designed with extra caution:
                         - The model is instructed to flag **any** forward movement, even slight drift.
                         - - It uses visual cues like black square size and road length to infer motion.
                           - - The model responds concisely with a verdict and single-sentence reasoning.
                            
                             
