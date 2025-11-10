"""Download e importação de modelos"""
import os
import subprocess
from huggingface_hub import hf_hub_download


class ModelImporter:
    
    TEMPLATES = {
        'mistral': '<s>[INST] {{ .Prompt }} [/INST]',
        'zephyr': '<|system|>\nYou are a helpful assistant.\n<|user|>\n{{ .Prompt }}\n<|assistant|>',
        'phi': 'Instruct: {{ .Prompt }}\nOutput:',
        'cinder': 'Instruct: {{ .Prompt }}\nOutput:',
        'gemma': '<start_of_turn>user\n{{ .Prompt }}\n<end_of_turn>\n<start_of_turn>model',
        'ezo': '<start_of_turn>user\n{{ .Prompt }}\n<end_of_turn>\n<start_of_turn>model'
    }
    
    @staticmethod
    def download_from_hf(repo_id, filename, destino="./modelos"):
        """Baixa modelo do HuggingFace"""
        try:
            os.makedirs(destino, exist_ok=True)
            print(f"Baixando {filename}...")
            
            caminho = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=destino,
                local_dir_use_symlinks=False
            )
            
            print(f"Download concluido: {caminho}")
            return caminho
        except Exception as e:
            print(f"Erro: {e}")
            return None
    
    @staticmethod
    def create_modelfile(caminho_gguf, nome_modelo):
        """Cria Modelfile"""
        caminho_abs = os.path.abspath(caminho_gguf)
        
        template = '{{ .Prompt }}'
        for key, tmpl in ModelImporter.TEMPLATES.items():
            if key in nome_modelo.lower():
                template = tmpl
                break
        
        content = f'''FROM {caminho_abs}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

TEMPLATE """{template}"""

SYSTEM """You are a helpful AI assistant."""
'''
        
        path = f"./Modelfile_{nome_modelo.replace('-', '_')}"
        with open(path, 'w') as f:
            f.write(content)
        
        return path
    
    @staticmethod
    def import_to_ollama(caminho_gguf, nome_ollama):
        """Importa para Ollama"""
        if not os.path.exists(caminho_gguf):
            return False
        
        modelfile = ModelImporter.create_modelfile(caminho_gguf, nome_ollama)
        
        cmd = ["ollama", "create", nome_ollama, "-f", modelfile]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        os.remove(modelfile)
        return result.returncode == 0