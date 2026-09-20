"""
Módulo Transformer (stub).

Reservado para reemplazar el backbone GRU por una arquitectura
Transformer cuando el dataset sea suficientemente grande.

Diseño previsto:
    - Encoder-only (estilo BERT) con heads multi-label.
    - Usar los módulos attention.py y embedding.py de este paquete.
    - Mantener la misma API de salida que SJGLanguageStructured:
        forward(input_ids) → dict de logits por campo semántico.
"""

# TODO: Implementar cuando el dataset supere los 2000 ejemplos.
