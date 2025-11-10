import pandas as pd
import glob

def consolidar_benchmarks():
    arquivos = sorted(glob.glob("benchmark/benchmark_*_cp*.csv"))

    if not arquivos:
        print("❌ Nenhum arquivo benchmark_*_cp*.csv encontrado no diretório atual.")
        return

    print(f"🔍 {len(arquivos)} arquivos encontrados. Consolidando resultados...")

    dfs = []
    for arq in arquivos:
        try:
            df_temp = pd.read_csv(arq)
            dfs.append(df_temp)
        except Exception as e:
            print(f"⚠️ Erro ao ler {arq}: {e}")

    # Concatenar e limpar duplicados
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp", "modelo", "pergunta"])

    # Calcular médias por modelo
    tabela = df.groupby("modelo").agg({
        "tempo_s": "mean",
        "tokens_por_segundo": "mean",
        "flops_por_segundo": "mean",
        "vram_usado_mb": "mean",
        "sucesso": lambda x: x.mean() * 100
    }).reset_index()

    # Renomear colunas para o formato final
    tabela = tabela.rename(columns={
        "modelo": "Modelo",
        "tempo_s": "Latência (s/tweet)",
        "tokens_por_segundo": "Throughput (tokens/s)",
        "flops_por_segundo": "FLOPs/s",
        "vram_usado_mb": "VRAM (MB)",
        "sucesso": "Sucesso (%)"
    })

    tabela = tabela.sort_values(by="Latência (s/tweet)").reset_index(drop=True)

    print("\n✅ Resultado consolidado:\n")
    print(tabela.round(2).to_string(index=False))
    tabela.to_csv("benchmark_resumo_final.csv", index=False)
    print("\n💾 Arquivo salvo como: benchmark_resumo_final.csv")

if __name__ == "__main__":
    consolidar_benchmarks()
