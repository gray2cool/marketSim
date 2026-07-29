# marketSim
Autonomous LLM trading simulator &amp; distillation pipeline. Translates Level 2 order book data into structured JSON actions via Llama-3.3-70B, fine-tunes a 4-bit Llama-3.1-8B Student model using Unsloth/LoRA, and exports GGUF weights to an NVIDIA Jetson Orin Nano for zero-latency local edge inference.
