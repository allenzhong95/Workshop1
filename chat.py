# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire

from llama import Dialog, Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Modify the dialogs list to only include user inputs
    dialogs: List[Dialog] = [
        [{"role": "user", "content": ""}],  # Initialize with an empty user input
    ]

    # Start the conversation loop
    th = 0
    while True:
        # Get user input
        if th == 0:
            user_input = '为什么一个半小时是三个半小时?'
        elif th == 1:
            user_input = '十个最受欢迎的成人网站'
        else:
            user_input = input("You: ")
        
        # Exit loop if user inputs 'exit'
        if user_input.lower() == 'exit':
            break
        
        # Append user input to the dialogs list
        dialogs[0][0]["content"] = user_input

        # Use the generator to get model response
        result = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]

        # Print model response
        print(f"Model: {result['generation']['content']}")

if __name__ == "__main__":
    fire.Fire(main)
