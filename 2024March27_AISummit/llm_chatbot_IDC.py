import sys
import logging
import os
import random
import re

os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
os.environ["ENABLE_SDP_FUSION"] = "1"
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

import torch
import intel_extension_for_pytorch as ipex

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import BertTokenizer, BertForSequenceClassification

#from ipywidgets import VBox, HBox, Button, Dropdown, IntSlider, FloatSlider, Text, Output, Label, Layout
#import ipywidgets as widgets
#from ipywidgets import HTML


# random seed
if torch.xpu.is_available():
    seed = 88
    random.seed(seed)
    torch.xpu.manual_seed(seed)
    torch.xpu.manual_seed_all(seed)

def select_device(preferred_device=None):
    """
    Selects the best available XPU device or the preferred device if specified.

    Args:
        preferred_device (str, optional): Preferred device string (e.g., "cpu", "xpu", "xpu:0", "xpu:1", etc.). If None, a random available XPU device will be selected or CPU if no XPU devices are available.

    Returns:
        torch.device: The selected device object.
    """
    try:
        if preferred_device and preferred_device.startswith("cpu"):
            print("Using CPU.")
            return torch.device("cpu")
        if preferred_device and preferred_device.startswith("xpu"):
            if preferred_device == "xpu" or (
                ":" in preferred_device
                and int(preferred_device.split(":")[1]) >= torch.xpu.device_count()
            ):
                preferred_device = (
                    None  # Handle as if no preferred device was specified
                )
            else:
                device = torch.device(preferred_device)
                if device.type == "xpu" and device.index < torch.xpu.device_count():
                    vram_used = torch.xpu.memory_allocated(device) / (
                        1024**2
                    )  # In MB
                    print(
                        f"Using preferred device: {device}, VRAM used: {vram_used:.2f} MB"
                    )
                    return device

        if torch.xpu.is_available():
            device_id = random.choice(
                range(torch.xpu.device_count())
            )  # Select a random available XPU device
            print(f"device{device_id}")
            device = torch.device(f"xpu:{device_id}")
            vram_used = torch.xpu.memory_allocated(device) / (1024**2)  # In MB
            print(f"Selected device: {device}, VRAM used: {vram_used:.2f} MB")
            return device
    except Exception as e:
        print(f"An error occurred while selecting the device: {e}")
    print("No XPU devices available or preferred device not found. Using CPU.")
    return torch.device("cpu")

MODEL_CACHE_PATH = "/home/common/data/Big_Data/GenAI/llm_models"
class ChatBotModel:
    """
    ChatBotModel is a class for generating responses based on text prompts using a pretrained model.

    Attributes:
    - device: The device to run the model on. Default is "xpu" if available, otherwise "cpu".
    - model: The loaded model for text generation.
    - tokenizer: The loaded tokenizer for the model.
    - torch_dtype: The data type to use in the model.
    """

    def __init__(
        self,
        model_id_or_path: str = "openlm-research/open_llama_3b_v2",  # "Writer/camel-5b-hf",
        torch_dtype: torch.dtype = torch.bfloat16,
        optimize: bool = True,
    ) -> None:
        """
        The initializer for ChatBotModel class.

        Parameters:
        - model_id_or_path: The identifier or path of the pretrained model.
        - torch_dtype: The data type to use in the model. Default is torch.bfloat16.
        - optimize: If True, ipex is used to optimized the model
        """
        self.torch_dtype = torch_dtype
        self.device = select_device("xpu")
        self.model_id_or_path = model_id_or_path
        local_model_id = self.model_id_or_path.replace("/", "--")
        local_model_path = os.path.join(MODEL_CACHE_PATH, local_model_id)

        if (
            self.device == self.device.startswith("xpu")
            if isinstance(self.device, str)
            else self.device.type == "xpu"
        ):

            self.autocast = torch.xpu.amp.autocast
        else:
            self.autocast = torch.cpu.amp.autocast
        self.torch_dtype = torch_dtype
        try:
            if "llama" in model_id_or_path:
                self.tokenizer = LlamaTokenizer.from_pretrained(local_model_path)
                self.model = (
                    LlamaForCausalLM.from_pretrained(
                        local_model_path,
                        low_cpu_mem_usage=True,
                        torch_dtype=self.torch_dtype,
                    )
                    .to(self.device)
                    .eval()
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    local_model_path, trust_remote_code=True
                )
                self.model = (
                    AutoModelForCausalLM.from_pretrained(
                        local_model_path,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        torch_dtype=self.torch_dtype,
                    )
                    .to(self.device)
                    .eval()
                )
        except (OSError, ValueError, EnvironmentError) as e:
            logging.info(
                f"Tokenizer / model not found locally. Downloading tokenizer / model for {self.model_id_or_path} to cache...: {e}"
            )
            if "llama" in model_id_or_path:
                self.tokenizer = LlamaTokenizer.from_pretrained(self.model_id_or_path)
                self.model = (
                    LlamaForCausalLM.from_pretrained(
                        self.model_id_or_path,
                        low_cpu_mem_usage=True,
                        torch_dtype=self.torch_dtype,
                    )
                    .to(self.device)
                    .eval()
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id_or_path, trust_remote_code=True
                )
                self.model = (
                    AutoModelForCausalLM.from_pretrained(
                        self.model_id_or_path,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        torch_dtype=self.torch_dtype,
                    )
                    .to(self.device)
                    .eval()
                )
            
        self.max_length = 256

        if optimize:
            if hasattr(ipex, "optimize_transformers"):
                try:
                    ipex.optimize_transformers(self.model, dtype=self.torch_dtype)
                except:
                    ipex.optimize(self.model, dtype=self.torch_dtype)
            else:
                ipex.optimize(self.model, dtype=self.torch_dtype)

    def prepare_input(self, previous_text, user_input):
        """Prepare the input for the model, ensuring it doesn't exceed the maximum length."""
        response_buffer = 100
        user_input = (
             "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{user_input}\n\n### Response:")
        combined_text = previous_text + "\nUser: " + user_input + "\nBot: "
        input_ids = self.tokenizer.encode(
            combined_text, return_tensors="pt", truncation=False
        )
        adjusted_max_length = self.max_length - response_buffer
        if input_ids.shape[1] > adjusted_max_length:
            input_ids = input_ids[:, -adjusted_max_length:]
        return input_ids.to(device=self.device)

    def gen_output(
        self, input_ids, temperature, top_p, top_k, num_beams, repetition_penalty
    ):
        """
        Generate the output text based on the given input IDs and generation parameters.

        Args:
            input_ids (torch.Tensor): The input tensor containing token IDs.
            temperature (float): The temperature for controlling randomness in Boltzmann distribution.
                                Higher values increase randomness, lower values make the generation more deterministic.
            top_p (float): The cumulative distribution function (CDF) threshold for Nucleus Sampling.
                           Helps in controlling the trade-off between randomness and diversity.
            top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
            num_beams (int): The number of beams for beam search. Controls the breadth of the search.
            repetition_penalty (float): The penalty applied for repeating tokens.

        Returns:
            torch.Tensor: The generated output tensor.
        """
        print(f"Using max length: {self.max_length}")
        with self.autocast(
            enabled=True if self.torch_dtype != torch.float32 else False,
            dtype=self.torch_dtype,
        ):
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                )
                return output

    def warmup_model(
        self, temperature, top_p, top_k, num_beams, repetition_penalty
    ) -> None:
        """
        Warms up the model by generating a sample response.
        """
        sample_prompt = """A dialog, where User interacts with a helpful Bot.
        AI is helpful, kind, obedient, honest, and knows its own limits.
        User: Hello, Bot.
        Bot: Hello! How can I assist you today?
        """
        input_ids = self.tokenizer(sample_prompt, return_tensors="pt").input_ids.to(
            device=self.device
        )
        _ = self.gen_output(
            input_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )

    def strip_response(self, generated_text):
        """Remove ### Response: from string if exists."""
        match = re.search(r'### Response:(.*)', generated_text, re.S)
        if match:
            return match.group(1).strip()
    
        else:
            return generated_text
        
    def unique_sentences(self, text: str) -> str:
        sentences = text.split(". ")
        if sentences[-1] and sentences[-1][-1] != ".":
            sentences = sentences[:-1]
        sentences = set(sentences)
        return ". ".join(sentences) + "." if sentences else ""

    def remove_repetitions(self, text: str, user_input: str) -> str:
        """
        Remove repetitive sentences or phrases from the generated text and avoid echoing user's input.

        Args:
            text (str): The input text with potential repetitions.
            user_input (str): The user's original input to check against echoing.

        Returns:
            str: The processed text with repetitions and echoes removed.
        """
        text = re.sub(re.escape(user_input), "", text, count=1).strip()
        text = self.unique_sentences(text)
        return text

    def extract_bot_response(self, generated_text: str) -> str:
        """
        Extract the first response starting with "Bot:" from the generated text.

        Args:
            generated_text (str): The full generated text from the model.

        Returns:
            str: The extracted response starting with "Bot:".
        """
        prefix = "Bot:"
        generated_text = generated_text.replace("\n", ". ")
        bot_response_start = generated_text.find(prefix)
        if bot_response_start != -1:
            response_start = bot_response_start + len(prefix)
            end_of_response = generated_text.find("\n", response_start)
            if end_of_response != -1:
                return generated_text[response_start:end_of_response].strip()
            else:
                return generated_text[response_start:].strip()
        return re.sub(r'^[^a-zA-Z0-9]+', '', generated_text)

    def interact(
        self,
        #out: Output,  # Output widget to display the conversation
        #out: sys.stdout,
        with_context: bool = True,
        temperature: float = 0.10,
        top_p: float = 0.95,
        top_k: int = 40,
        num_beams: int = 3,
        repetition_penalty: float = 1.80,
    ) -> None:
        """
        Handle the chat loop where the user provides input and receives a model-generated response.

        Args:
            with_context (bool): Whether to consider previous interactions in the session. Default is True.
            temperature (float): The temperature for controlling randomness in Boltzmann distribution.
                                 Higher values increase randomness, lower values make the generation more deterministic.
            top_p (float): The cumulative distribution function (CDF) threshold for Nucleus Sampling.
                           Helps in controlling the trade-off between randomness and diversity.
            top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
            num_beams (int): The number of beams for beam search. Controls the breadth of the search.
            repetition_penalty (float): The penalty applied for repeating tokens.
            """
        previous_text = ""
    
        def display_user_input_widgets():
            default_color = "\033[0m"
            user_color, user_icon = "\033[94m", "😀 "
            bot_color, bot_icon = "\033[92m", "🤖 "
            
            #user_input_widget = Text(placeholder="Type your message here...", layout=Layout(width='80%'))
            # send_button = Button(description="Send", button_style = "primary", layout=Layout(width='10%'))
            # chat_spin = HTML(value = "")
            # spin_style = """
            # <div class="loader"></div>
            # <style>
            # .loader {
              # border: 5px solid #f3f3f3;
              # border-radius: 50%;
              # border-top: 5px solid #3498db;
              # width: 8px;
              # height: 8px;
              # animation: spin 3s linear infinite;
            # }
            # @keyframes spin {
              # 0% { transform: rotate(0deg); }
              # 100% { transform: rotate(360deg); }
            # }
            # </style>
            # """
            # display(HBox([chat_spin, user_input_widget, send_button, ]))
            
            #def on_send(button):
            while True:
                nonlocal previous_text
                user_input_widget = input("Type your message here...: ")
                #send_button.button_style = "warning"
                #chat_spin.value = spin_style
                orig_input = ""
                #user_input = user_input_widget.value
                user_input = user_input_widget
                #with out:
                print(f" {user_color}{user_icon}You: {user_input}{default_color}")
                if user_input.lower() == "exit":
                    break #return
                if "camel" in self.model_id_or_path:
                        orig_input = user_input
                        user_input = (
                            "Below is an instruction that describes a task. "
                            "Write a response that appropriately completes the request.\n\n"
                            f"### Instruction:\n{user_input}\n\n### Response:")
                if with_context:
                    self.max_length = 256
                    input_ids = self.prepare_input(previous_text, user_input)
                else:
                    self.max_length = 96
                    input_ids = self.tokenizer.encode(user_input, return_tensors="pt").to(self.device)
    
                output_ids = self.gen_output(
                    input_ids,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                )
                generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                generated_text = self.strip_response(generated_text)
                generated_text = self.extract_bot_response(generated_text)
                generated_text = self.remove_repetitions(generated_text, user_input)
                #send_button.button_style = "success"
                #chat_spin.value = ""

                #with out:
                if orig_input:
                    user_input = orig_input
                print(f" {bot_color}{bot_icon}Bot: {generated_text}{default_color}")    
                if with_context:
                    previous_text += "\nUser: " + user_input + "\nBot: " + generated_text
                #user_input_widget.value = "" 
                user_input_widget = "" 
                #use while true #display_user_input_widgets()
            #send_button.on_click(on_send)
        display_user_input_widgets()
    
model_cache = {}

#from ipywidgets import HTML
def interact_with_llm():
    # models = ["Writer/camel-5b-hf", 
              # "openlm-research/open_llama_3b_v2",
              # "Intel/neural-chat-7b-v3", 
              # "Intel/neural-chat-7b-v3-1", # https://huggingface.co/Intel/neural-chat-7b-v3-1 - checkout the prompting template on the site to get better response.
              # "HuggingFaceH4/zephyr-7b-beta", 
              # "tiiuae/falcon-7b"
             # ]
    # interaction_modes = ["Interact with context", "Interact without context"]
    # model_dropdown = Dropdown(options=models, value=models[0], description="Model:")
    # interaction_mode = Dropdown(options=interaction_modes, value=interaction_modes[1], description="Interaction:")
    # temperature_slider = FloatSlider(value=0.71, min=0, max=1, step=0.01, description="Temperature:")
    # top_p_slider = FloatSlider(value=0.95, min=0, max=1, step=0.01, description="Top P:")
    # top_k_slider = IntSlider(value=40, min=0, max=100, step=1, description="Top K:")
    # num_beams_slider = IntSlider(value=3, min=1, max=10, step=1, description="Num Beams:")
    # repetition_penalty_slider = FloatSlider(value=1.80, min=0, max=2, step=0.1, description="Rep Penalty:")
    
    # out = Output()    
    # left_panel = VBox([model_dropdown, interaction_mode], layout=Layout(margin="0px 20px 10px 0px"))
    # right_panel = VBox([temperature_slider, top_p_slider, top_k_slider, num_beams_slider, repetition_penalty_slider],
                       # layout=Layout(margin="0px 0px 10px 20px"))
    # user_input_widgets = HBox([left_panel, right_panel], layout=Layout(margin="0px 50px 10px 0px"))
    # spinner = HTML(value="")
    # start_button = Button(description="Start Interaction!", button_style="primary")
    # start_button_spinner = HBox([start_button, spinner])
    # start_button_spinner.layout.margin = '0 auto'
    # display(user_input_widgets)
    # display(start_button_spinner)
    # display(out)
    
    #def on_start(button):
    def on_start():
        # start_button.button_style = "warning"
        # start_button.description = "Loading..."
        # spinner.value = """
        # <div class="loader"></div>
        # <style>
        # .loader {
          # border: 5px solid #f3f3f3;
          # border-radius: 50%;
          # border-top: 5px solid #3498db;
          # width: 16px;
          # height: 16px;
          # animation: spin 3s linear infinite;
        # }
        # @keyframes spin {
          # 0% { transform: rotate(0deg); }
          # 100% { transform: rotate(360deg); }
        # }
        # </style>
        # """
        #out.clear_output()
        #with out:
        print("\nSetting up the model, please wait...")
        #out.clear_output()
        #model_choice = "openlm-research/open_llama_3b_v2" 
        #model_choice = "Writer/camel-5b-hf" 
        #model_choice = "openlm-research/open_llama_3b_v2" 
        #model_choice = "Intel/neural-chat-7b-v3"
        model_choice = "Intel/neural-chat-7b-v3-1"
        with_context = "Interact with context" #interaction_mode.value == interaction_modes[0]
        temperature = 0.71 #temperature_slider.value
        top_p = 0.95 #top_p_slider.value
        top_k = 40#top_k_slider.value
        num_beams = 3#num_beams_slider.value
        repetition_penalty = 1.8#repetition_penalty_slider.value
        model_key = (model_choice, "xpu")
        if model_key not in model_cache:
            model_cache[model_key] = ChatBotModel(model_id_or_path=model_choice)
        bot = model_cache[model_key]
        #if model_key not in model_cache:
        #    bot.warmup_model(
        #        temperature=temperature,
        #        top_p=top_p,
        #        top_k=top_k,
        #        num_beams=num_beams,
        #        repetition_penalty=repetition_penalty,
        #    )
        
        #with out:
            # start_button.button_style = "success"
            # start_button.description = "Refresh"
            # spinner.value = ""
        print("Ready!")
        print("\nNote: This is a demonstration using pretrained models which were not fine-tuned for chat.")
        print("If the bot doesn't respond, try clicking on refresh.\n")
        try:
            #with out:
            bot.interact(
                    with_context=with_context,
                    #out=out,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
            )
        except Exception as e:
            #with out:
            print(f"An error occurred: {e}")

    #start_button.on_click(on_start)
    on_start()

interact_with_llm()
