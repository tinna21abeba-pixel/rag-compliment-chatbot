"""
generator.py
------------
Handles prompt construction and LLM-based answer generation.

Supports two backends:
  1. HuggingFace pipeline (local models: Mistral-7B, Llama-3, Falcon, etc.)
  2. LangChain HuggingFaceHub (API-based, no local GPU required)

The prompt template is carefully designed to:
  - Ground the model strictly in retrieved complaint excerpts
  - Prevent hallucination by explicitly instructing "use only the context"
  - Encourage structured, professional analyst-style answers
  - Handle the "I don't know" case gracefully
"""

from __future__ import annotations

import logging
import os
from typing import Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a financial analyst assistant for CrediTrust Financial, \
a digital finance company serving East African markets. Your role is to help \
internal teams — product managers, compliance officers, and customer support leads — \
understand patterns in customer complaints.

Rules you MUST follow:
1. Base your answer ONLY on the complaint excerpts provided in the Context section.
2. If the context does not contain enough information to answer the question, \
state clearly: "The available complaint data does not contain sufficient information \
to answer this question."
3. Do NOT fabricate complaints, statistics, or company names.
4. Be concise and professional. Use bullet points when listing multiple issues.
5. When relevant, mention the product category or company name from the sources.
"""

PROMPT_TEMPLATE = """{system}

Context (retrieved complaint excerpts):
-----------------------------------------
{context}
-----------------------------------------

Question: {question}

Answer:"""


def build_prompt(context_chunks: list, question: str) -> str:
    """
    Build the full prompt string by injecting retrieved chunks into the template.

    Parameters
    ----------
    context_chunks : list[RetrievedChunk]
        Top-k chunks returned by the retriever.
    question : str
        The user's question.

    Returns
    -------
    str
        Fully formatted prompt ready for the LLM.
    """
    if not context_chunks:
        context_str = "No relevant complaint excerpts were retrieved."
    else:
        parts = []
        for i, chunk in enumerate(context_chunks, start=1):
            parts.append(
                f"[Source {i} | ID:{chunk.complaint_id} | "
                f"{chunk.product_category} | {chunk.issue}]\n{chunk.text}"
            )
        context_str = "\n\n".join(parts)

    return PROMPT_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context_str,
        question=question,
    )


# ---------------------------------------------------------------------------
# HuggingFace Pipeline generator (local)
# ---------------------------------------------------------------------------
class HFPipelineGenerator:
    """
    Generates answers using a locally loaded HuggingFace text-generation model.

    Suitable for: Mistral-7B-Instruct, Llama-3-8B-Instruct, Falcon-7B-Instruct.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID.
    max_new_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature. Lower = more deterministic.
    device : int
        GPU device index. -1 for CPU.
    load_in_4bit : bool
        Whether to load in 4-bit quantization (requires bitsandbytes).
    """

    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        device: int = -1,
        load_in_4bit: bool = False,
    ) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.info("Loading LLM: %s (4bit=%s)", model_id, load_in_4bit)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb_config, device_map="auto"
            )
            device_arg = None
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id)
            device_arg = device

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            device=device_arg,
            return_full_text=False,  # return only the generated part
        )
        logger.info("LLM pipeline ready.")

    def generate(self, prompt: str) -> str:
        """
        Generate an answer for the given prompt.

        Parameters
        ----------
        prompt : str
            Fully formatted prompt (from build_prompt).

        Returns
        -------
        str
            Generated answer text.
        """
        outputs = self.pipe(prompt)
        answer = outputs[0]["generated_text"].strip()
        return answer


# ---------------------------------------------------------------------------
# HuggingFace Hub generator (API-based, no local GPU required)
# ---------------------------------------------------------------------------
class HFHubGenerator:
    """
    Generates answers via the HuggingFace Inference API.

    Suitable when local GPU is unavailable. Requires a HF_TOKEN env variable.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID (must be inference-API enabled).
    max_new_tokens : int
    temperature : float
    """

    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is not installed. Run: pip install huggingface_hub"
            ) from exc

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning(
                "HF_TOKEN environment variable not set. "
                "Some models may be rate-limited or unavailable."
            )

        self.client = InferenceClient(model=model_id, token=hf_token)
        self.max_new_tokens = max_new_tokens
        self.temperature = max(temperature, 0.01)  # API requires > 0
        logger.info("HFHub generator initialised for model: %s", model_id)

    def generate(self, prompt: str) -> str:
        """
        Call the HuggingFace Inference API and return the generated answer.
        """
        response = self.client.text_generation(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            stream=False,
        )
        return response.strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_generator(
    backend: str = "hf_hub",
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    device: int = -1,
    load_in_4bit: bool = False,
):
    """
    Factory that returns the appropriate generator.

    Parameters
    ----------
    backend : str
        'hf_pipeline' (local) or 'hf_hub' (API-based).
    """
    backend = backend.lower().strip()
    if backend == "hf_pipeline":
        return HFPipelineGenerator(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            load_in_4bit=load_in_4bit,
        )
    elif backend == "hf_hub":
        return HFHubGenerator(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unknown generator backend '{backend}'. "
            "Choose 'hf_pipeline' or 'hf_hub'."
        )
