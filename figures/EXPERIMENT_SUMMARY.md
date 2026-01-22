# Reporte de Validación - Respuesta a Revisor
**Fecha:** 2026-01-21T23:10:42.919959
**Checkpoint usado:** /home/grisun0/src/py/strass/github/checkpoints/strassen_exact.pt

## Verificación de Grokking
- **Accuracy:** 100.00%
- **LC:** 1.000000
- **Slots activos:** 7
- **Error máximo:** 1.67e-06
- **Error medio:** 7.82e-08

## Resultados de Poda Iterativa
- **Sparsity final:** 0.00%
- **Accuracy final:** 100.00%
- **Cuenca discreta alcanzada:** Sí
- **Éxito del protocolo:** No
- **Razón de parada:** delta=100.0000 > threshold=0.1
- **Iteraciones:** 11

## Análisis ROC/AUC (Modelo Individual)
- **AUC:** Indefinido (Single class - cannot compute ROC/AUC)
- **Distribución de clases:** {'class_0': 0, 'class_1': 10000}

## Experimento LC desde Cero (Punto B)
- **Época de grokking:** 2160
- **Accuracy final:** 100.00%
- **LC rango:** [0.0000, 441.5850]

## Runs Balanceados (Punto C)
- **Total runs:** 60
- **Grokked:** 28
- **No grokkeado:** 32
- **AUC (balanceado):** 1.0000
- **IC 95%:** [1.0000, 1.0000]

## Archivos Generados
- figure2_pruning_results.png
- experiment_results.json
- EXPERIMENT_SUMMARY.md
- figure3_roc_curves.png
- figure4_roc_balanced_runs.pdf
- figure4b_balanced_runs_summary.pdf
- figure1b_lc_training.png
- figure4_roc_balanced_runs.png
- figure2_pruning_results.pdf
- figure4b_balanced_runs_summary.png
- figure1b_lc_training.pdf
- figure3_roc_curves.pdf
