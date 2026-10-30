# Función auxiliar (sin prefijo test_)
def obtener_texto():
    return "tested"

# Esta es la única prueba real
def test_test():
    assert obtener_texto() == "tested"
