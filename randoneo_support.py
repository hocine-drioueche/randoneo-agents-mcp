"""
Serveur MCP du support Randoneo.

Expose 4 tools et 1 resource :
- search_catalog : recherche au catalogue
- get_order_status : état d'une commande
- search_knowledge_base : recherche documentaire (RAG)
- create_ticket : escalade vers un humain
- randoneo://product/{sku} : fiche produit
"""

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from fastmcp import FastMCP

# Charge le .env
load_dotenv()

# Crée le serveur MCP
mcp = FastMCP("randoneo-support")

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

DATA_DIR = Path(__file__).parent / "data"

CATALOG = json.loads((DATA_DIR / "catalog.json").read_text(encoding="utf-8"))
ORDERS = json.loads((DATA_DIR / "orders.json").read_text(encoding="utf-8"))
CORPUS = json.loads((DATA_DIR / "randoneo_corpus.json").read_text(encoding="utf-8"))

# Stockage en mémoire des tickets
TICKETS = []

# ============================================================
# BASE VECTORIELLE (RAG)
# ============================================================

embed = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)

knowledge_base = chromadb.Client().create_collection(
    name="randoneo_kb",
    embedding_function=embed,
    metadata={"hnsw:space": "cosine"},
)

knowledge_base.add(
    ids=list(CORPUS.keys()),
    documents=[entry["text"] for entry in CORPUS.values()],
)


# ============================================================
# TOOL 1 : search_catalog
# ============================================================

@mcp.tool
def search_catalog(query: str, category: str | None = None) -> list[dict]:
    """Recherche des produits au catalogue Randoneo par mots-clés.
    
    À utiliser dès qu'un client demande un prix, une disponibilité ou un produit.
    
    Args:
        query: mots-clés du client, par ex. « tente légère 2 places ».
        category: catégorie à laquelle restreindre la recherche (Tentes, Sacs, Chaussures…).
    
    Renvoie une liste de produits {sku, name, price_eur, stock}.
    """
    terms = [t for t in query.lower().split() if len(t) > 2]
    found = []
    
    for item in CATALOG:
        # Filtre par catégorie
        if category and item["category"].lower() != category.lower():
            continue
        
        # Cherche dans name + description + category
        haystack = f"{item['name']} {item['description']} {item['category']}".lower()
        score = sum(haystack.count(term) for term in terms)
        
        if score:
            found.append((score, {
                "sku": item["sku"],
                "name": item["name"],
                "price_eur": item["price_eur"],
                "stock": item["stock"],
            }))
    
    found.sort(key=lambda pair: -pair[0])
    return [product for _, product in found[:5]]


# ============================================================
# TOOL 2 : get_order_status
# ============================================================

@mcp.tool
def get_order_status(order_id: str) -> dict:
    """Donne l'état d'une commande Randoneo à partir de son identifiant.
    
    À utiliser dès qu'un client demande où en est sa commande.
    
    Args:
        order_id: identifiant de commande, par ex. « RND-10238 ».
    """
    for order in ORDERS:
        if order["order_id"].upper() == order_id.strip().upper():
            return {
                "order_id": order["order_id"],
                "status": order["status"],
                "shipping_method": order["shipping_method"],
                "tracking_number": order["tracking_number"] or "Non disponible",
                "eta": order["eta"],
            }
    return {"erreur": f"Commande « {order_id} » introuvable. Format attendu : RND-XXXXX."}


# ============================================================
# TOOL 3 : search_knowledge_base
# ============================================================

@mcp.tool
def search_knowledge_base(query: str) -> list[str]:
    """Cherche dans la documentation Randoneo.
    
    À utiliser pour les politiques (retour, remboursement, livraison, paiement,
    garantie) et les informations produit.
    
    Args:
        query: la question ou les mots-clés à rechercher.
    """
    result = knowledge_base.query(query_texts=[query], n_results=3)
    return result["documents"][0]


# ============================================================
# TOOL 4 : create_ticket
# ============================================================

@mcp.tool
def create_ticket(intent: str, summary: str, order_id: str | None = None) -> str:
    """Ouvre un ticket au support Randoneo pour escalader une demande vers un humain.
    
    Action sensible : à utiliser uniquement quand le client demande une action
    (remplacement, remboursement, réclamation).
    
    Args:
        intent: nature de la demande (return_refund, payment_issue, order_tracking, account).
        summary: résumé de la demande du client, en une phrase.
        order_id: identifiant de la commande concernée (optionnel).
    """
    TICKETS.append({
        "intent": intent,
        "summary": summary,
        "order_id": order_id,
    })
    return f"Ticket TICK-{len(TICKETS):03d} ouvert ({intent})."


# ============================================================
# RESOURCE : fiche produit
# ============================================================

@mcp.resource("randoneo://product/{sku}")
def product_sheet(sku: str) -> str:
    """Fiche détaillée d'un produit Randoneo, au format Markdown."""
    product = next((p for p in CATALOG if p["sku"].upper() == sku.strip().upper()), None)
    
    if product is None:
        return f"# Produit {sku} introuvable\n\nVérifiez le SKU."
    
    lines = [
        f"# {product['name']}",
        f"- **SKU** : {product['sku']}",
        f"- **Catégorie** : {product['category']}",
        f"- **Prix** : {product['price_eur']} €",
        f"- **Stock** : {product['stock']}",
        "",
        f"## Description",
        product["description"],
        "",
        "## Attributs",
    ]
    
    for key, value in product.get("attributes", {}).items():
        lines.append(f"- **{key}** : {value}")
    
    return "\n".join(lines)


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    mcp.run()  # stdio par défaut