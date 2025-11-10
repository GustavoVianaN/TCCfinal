"""Monitoramento de GPU via nvidia-smi"""
import subprocess


class GPUMonitor:
    
    @staticmethod
    def get_vram_info(limite_percent=75):
        """Obtém informações de VRAM"""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=5, encoding='utf-8', errors='ignore'
            )
            
            if result.returncode == 0:
                valores = result.stdout.strip().split(',')
                
                if len(valores) >= 2:
                    usado_mb = int(valores[0].strip())
                    total_mb = int(valores[1].strip())
                    percent_usado = (usado_mb / total_mb) * 100
                    
                    return {
                        'usado_mb': usado_mb,
                        'total_mb': total_mb,
                        'percent_usado': percent_usado,
                        'livre_mb': total_mb - usado_mb,
                        'disponivel': percent_usado < limite_percent,
                        'critico': percent_usado > 90,
                        'gpu_utilizacao': int(valores[2].strip()) if len(valores) > 2 else 0,
                        'temperatura': int(valores[3].strip()) if len(valores) > 3 else 0
                    }
            
            return {'erro': 'Formato inesperado'}
            
        except subprocess.TimeoutExpired:
            return {'erro': 'nvidia-smi timeout'}
        except FileNotFoundError:
            return {'erro': 'nvidia-smi não encontrado'}
        except Exception as e:
            return {'erro': str(e)}
    
    @staticmethod
    def print_status(vram_info, vram_estimado=None):
        """Imprime status formatado"""
        if 'erro' in vram_info:
            print(f"VRAM: {vram_info['erro']}")
            return False
        
        status = "[CRITICO]" if vram_info.get('critico') else "[OK]" if vram_info['disponivel'] else "[ALTO]"
        
        info = f"{status} VRAM: {vram_info['usado_mb']:,}MB/{vram_info['total_mb']:,}MB ({vram_info['percent_usado']:.1f}%)"
        
        if vram_info.get('temperatura', 0) > 0:
            info += f" | Temp: {vram_info['temperatura']}C"
        
        print(info)
        
        if vram_info.get('critico'):
            print("Aviso: VRAM critica! Use opcao 5")
            return False
        
        return vram_info['disponivel']