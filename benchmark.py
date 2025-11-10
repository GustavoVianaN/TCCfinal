"""Executor de benchmarks"""
import time
from datetime import datetime
import pandas as pd
from gpu_monitor import GPUMonitor
from ollama_manager import OllamaManager
from model_importer import ModelImporter
from configurations import MODELS_CONFIG, CHECKPOINT_INTERVAL


class BenchmarkExecutor:
    
    def __init__(self, limite_vram=75):
        self.limite_vram = limite_vram
        self.resultados = []
        self.models_config = MODELS_CONFIG
    
    def prepare_models(self):
        """Prepara modelos"""
        print("Preparando modelos...")
        prontos = []
        
        for i, config in enumerate(self.models_config, 1):
            nome = config['ollama_name']
            print(f"\n[{i}/5] {nome}")
            
            existe, _ = OllamaManager.model_exists(nome)
            if existe:
                print(f"Ja existe: {nome}")
                prontos.append(nome)
                continue
            
            caminho = ModelImporter.download_from_hf(config['repo_id'], config['filename'])
            
            if caminho and ModelImporter.import_to_ollama(caminho, nome):
                prontos.append(nome)
        
        print(f"\nProntos: {len(prontos)}/5")
        return prontos
    
    def execute_test(self, modelo, pergunta, timeout=600):
        """Executa teste individual"""
        vram_antes = GPUMonitor.get_vram_info()
        inicio = time.perf_counter()
        
        try:
            result = OllamaManager.run_model(modelo, pergunta, timeout)
            fim = time.perf_counter()
            vram_depois = GPUMonitor.get_vram_info()
            
            if result.returncode == 0:
                resposta = result.stdout.strip()
                tempo = fim - inicio
                
                tokens_entrada = len(pergunta.split()) * 1.3
                tokens_saida = len(resposta.split()) * 1.3
                tokens_por_seg = tokens_saida / tempo if tempo > 0 else 0
                
                # Busca parâmetros do modelo
                params = next((c['params'] for c in self.models_config if c['ollama_name'] == modelo), 7e9)
                
                flops_total = params * 2 * tokens_saida
                flops_por_seg = flops_total / tempo if tempo > 0 else 0
                
                return {
                    'timestamp': datetime.now(),
                    'modelo': modelo,
                    'pergunta': pergunta,
                    'resposta': resposta[:300],
                    'sucesso': True,
                    'tempo_s': round(tempo, 3),
                    'tokens_por_segundo': round(tokens_por_seg, 2),
                    'flops_por_segundo': int(flops_por_seg),
                    'vram_usado_mb': vram_depois.get('usado_mb', 0),
                    'erro': None
                }
            
            return {'sucesso': False, 'erro': result.stderr}
            
        except Exception as e:
            return {'sucesso': False, 'erro': str(e)}
    
    def save_checkpoint(self, sufixo=""):
        """Salva checkpoint"""
        if not self.resultados:
            return
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        arquivo = f"benchmark_{ts}{sufixo}.csv"
        
        pd.DataFrame(self.resultados).to_csv(arquivo, index=False, encoding='utf-8')
        print(f"Checkpoint: {arquivo}")
    
    def run_benchmark(self, perguntas, timeout=600):
        """Executa benchmark completo"""
        OllamaManager.stop_all_models()
        
        modelos = [c['ollama_name'] for c in self.models_config]
        total = len(modelos) * len(perguntas)
        contador = 0
        checkpoint_num = 0
        
        print(f"Total: {total} testes")
        
        for modelo in modelos:
            existe, nome_real = OllamaManager.model_exists(modelo)
            if not existe:
                continue
            
            for pergunta in perguntas:
                contador += 1
                print(f"\n[{contador}/{total}] {nome_real}")
                pergunta = f"""Classify the emotion expressed in the following tweet. Respond with ONLY one word: anger, joy, sadness, or optimism

Tweet: {pergunta}

Emotion:"""
                resultado = self.execute_test(nome_real, pergunta, timeout)
                self.resultados.append(resultado)
                
                if contador % CHECKPOINT_INTERVAL == 0:
                    checkpoint_num += 1
                    self.save_checkpoint(f"_cp{checkpoint_num}")
            
            OllamaManager.stop_all_models()
        
        return pd.DataFrame(self.resultados)