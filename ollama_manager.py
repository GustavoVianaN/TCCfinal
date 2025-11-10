"""Gerenciamento de operações do Ollama"""
import subprocess
import time


class OllamaManager:
    
    @staticmethod
    def list_models():
        """Lista modelos disponíveis"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                linhas = result.stdout.strip().split('\n')
                return [linha.split()[0] for linha in linhas[1:] if linha.strip()]
            
            return []
        except Exception:
            return []
    
    @staticmethod
    def model_exists(nome_modelo):
        """Verifica se modelo existe"""
        modelos = OllamaManager.list_models()
        
        if nome_modelo in modelos:
            return True, nome_modelo
        
        for modelo in modelos:
            if nome_modelo.lower() in modelo.lower():
                return True, modelo
        
        return False, None
    
    @staticmethod
    def stop_all_models():
        """Para todos os modelos"""
        try:
            result = subprocess.run(["ollama", "ps"], capture_output=True, text=True)
            
            if result.returncode == 0:
                for linha in result.stdout.strip().split('\n')[1:]:
                    if linha.strip():
                        modelo = linha.split()[0]
                        subprocess.run(["ollama", "stop", modelo], capture_output=True)
                
                time.sleep(3)
                return True
            
            return False
        except Exception:
            return False
    
    @staticmethod
    def run_model(modelo, prompt, timeout=600):
        """Executa modelo com prompt"""
        cmd = ["ollama", "run", modelo, prompt]
        return subprocess.run(
            cmd, capture_output=True, text=True,
            encoding='utf-8', timeout=timeout
        )