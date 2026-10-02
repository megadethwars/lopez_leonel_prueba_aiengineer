"""Base de conocimiento entregada por el cliente.

Los textos se copian tal cual (no editar). Cada documento es corto (1 párrafo),
por lo que se indexa como un único chunk; ver SUBMISSION.md.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    content: str


DOCUMENTS: list[Document] = [
    Document(
        doc_id="doc1",
        title="Política de garantía",
        content=(
            "Todos los electrodomésticos grandes (refrigeradoras, lavadoras, estufas) tienen "
            "garantía de 12 meses desde la fecha de compra. Electrodomésticos pequeños "
            "(licuadoras, planchas, tostadoras) tienen garantía de 6 meses. La garantía cubre "
            "defectos de fábrica, no daños por mal uso."
        ),
    ),
    Document(
        doc_id="doc2",
        title="Política de devoluciones",
        content=(
            "Los productos pueden devolverse dentro de 30 días de la compra si están sin usar y "
            "en su empaque original. Devoluciones después de 30 días solo se aceptan si el "
            "producto tiene un defecto cubierto por garantía. No se aceptan devoluciones de "
            "productos personalizados o en oferta final (“liquidación”)."
        ),
    ),
    Document(
        doc_id="doc3",
        title="Tiempos de envío",
        content=(
            "Envíos a la capital: 2-3 días hábiles. Envíos a otras ciudades: 5-7 días hábiles. "
            "Envíos internacionales no están disponibles actualmente."
        ),
    ),
    Document(
        doc_id="doc4",
        title="Reembolsos",
        content=(
            "Los reembolsos se procesan en 5-10 días hábiles después de recibir el producto "
            "devuelto. Se reembolsa al mismo método de pago original. Reembolsos mayores a $500 "
            "requieren aprobación de un supervisor humano — el agente no debe aprobarlos "
            "automáticamente."
        ),
    ),
    Document(
        doc_id="doc5",
        title="Canales de contacto",
        content=(
            "Para quejas sobre el trato de un empleado, disputas de facturación, o cualquier tema "
            "legal, el cliente debe ser referido a un agente humano en soporte@tiendahogar.example "
            "— el asistente de IA no debe intentar resolver estos casos."
        ),
    ),
]

HUMAN_SUPPORT_EMAIL = "soporte@tiendahogar.example"
