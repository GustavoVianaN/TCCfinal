"""Configurações e constantes do projeto"""
import pandas as pd

MODELS_CONFIG = [
    {
        'repo_id': 'tensorblock/Mistral-14b-Merge-Base-GGUF',
        'filename': 'Mistral-14b-Merge-Base-Q2_K.gguf',
        'ollama_name': 'mistral-14b-merge',
        'vram_mb': 7000,
        'params': 14_000_000_000
    },
    {
        'repo_id': 'TheBloke/mistral-7B-finetuned-orca-dpo-v2-GGUF',
        'filename': 'mistral-7b-finetuned-orca-dpo-v2.Q2_K.gguf',
        'ollama_name': 'mistral-7b-orca-dpo',
        'vram_mb': 3500,
        'params': 7_000_000_000
    },
    {
        'repo_id': 'RichardErkhov/interview-eval_-_zephyr-7b-stem-case-1-gguf',
        'filename': 'zephyr-7b-stem-case-1.Q8_0.gguf',
        'ollama_name': 'zephyr-7b-stem',
        'vram_mb': 5500,
        'params': 7_000_000_000
    },
    {
        'repo_id': 'Josephgflowers/Cinder-Phi-2-STEM-2.94B-Test',
        'filename': 'Cinder-Phi-2-Test.f16.gguf',
        'ollama_name': 'cinder-phi-2-stem',
        'vram_mb': 2000,
        'params': 2_700_000_000
    },
    {
        'repo_id': 'tensorblock/AXCXEPT_EZO-Humanities-9B-gemma-2-it-GGUF',
        'filename': 'EZO-Humanities-9B-gemma-2-it-Q2_K.gguf',
        'ollama_name': 'ezo-humanities-9b',
        'vram_mb': 4500,
        'params': 9_000_000_000
    }
]

BENCHMARK_QUESTIONS = pd.read_csv("balanced_emotion_data.csv").text.to_list()

VRAM_LIMIT_PERCENT = 75
VRAM_CRITICAL_PERCENT = 90
DEFAULT_TIMEOUT = 600
CHECKPOINT_INTERVAL = 10