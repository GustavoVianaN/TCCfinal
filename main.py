"""Arquivo principal"""
from benchmark import BenchmarkExecutor
from gpu_monitor import GPUMonitor
from ollama_manager import OllamaManager
from configurations import BENCHMARK_QUESTIONS, VRAM_LIMIT_PERCENT


def main():
    benchmark = BenchmarkExecutor(limite_vram=VRAM_LIMIT_PERCENT)
    
    print("Benchmark LLM")
    print("=" * 40)
    
    vram = GPUMonitor.get_vram_info(VRAM_LIMIT_PERCENT)
    GPUMonitor.print_status(vram)
    
    print("\nOpcoes:")
    print("1. Preparar modelos")
    print("2. Executar benchmark")
    print("3. Preparar E executar")
    print("4. Verificar modelos")
    print("5. Limpar VRAM")
    
    opcao = input("\nEscolha (1-5): ")
    
    if opcao == "1":
        benchmark.prepare_models()
    
    elif opcao == "2":
        df = benchmark.run_benchmark(BENCHMARK_QUESTIONS)
        print(f"\nConcluido: {len(df)} registros")
    
    elif opcao == "3":
        benchmark.prepare_models()
        df = benchmark.run_benchmark(BENCHMARK_QUESTIONS)
        print(f"\nConcluido: {len(df)} registros")
    
    elif opcao == "4":
        modelos = OllamaManager.list_models()
        print(f"\nModelos: {len(modelos)}")
        for m in modelos:
            print(f"  - {m}")
    
    elif opcao == "5":
        OllamaManager.stop_all_models()
        vram = GPUMonitor.get_vram_info(VRAM_LIMIT_PERCENT)
        GPUMonitor.print_status(vram)


if __name__ == "__main__":
    main()