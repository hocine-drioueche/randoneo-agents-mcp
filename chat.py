"""
Boucle de chat interactive pour l'assistant Randoneo.

- Streaming des étapes (outil appelé, arguments, résultat) puis des tokens
- Validation humaine avant toute action sensible (interrupt / Command(resume))
- Mémoire persistante : relancer avec --thread <id> reprend la conversation

L'agent (modèle, découverte MCP, garde-fou) vit dans agent.py : ce fichier
ne fait que piloter la conversation.
"""

import argparse
import asyncio
import json
import uuid

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from agent import open_agent

DB = "randoneo_memory.sqlite"


# ============================================================
# OUTILS D'AFFICHAGE
# ============================================================

def texte(content) -> str:
    """Extrait le texte d'un contenu de message (str ou liste de blocs)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def court(valeur, limite=140) -> str:
    s = valeur if isinstance(valeur, str) else json.dumps(valeur, ensure_ascii=False)
    s = " ".join(s.split())
    return s if len(s) <= limite else s[: limite - 1] + "…"


def demander_validation(demande: dict) -> bool:
    """Présente l'action prévue à l'opérateur et lit sa décision."""
    print("\n" + "=" * 60)
    print("  ⚠️  VALIDATION HUMAINE REQUISE")
    print("=" * 60)
    print(f"  Action : {demande['action']}")
    print("  Arguments :")
    for cle, valeur in demande["args"].items():
        print(f"    - {cle} : {valeur}")
    print("=" * 60)

    while True:
        reponse = input("\n  Approuver ? (o/n) : ").strip().lower()
        if reponse in ("o", "oui", "y", "yes"):
            return True
        if reponse in ("n", "non", "no"):
            return False
        print("  Répondez par 'o' ou 'n'.")


# ============================================================
# UN TOUR DE CONVERSATION (avec streaming)
# ============================================================

async def tour(agent, entree, config):
    """Fait avancer l'agent d'un cran en diffusant étapes et tokens.

    Retourne la demande de validation (dict) si le graphe s'est interrompu,
    sinon None une fois la réponse finale affichée.
    """
    flux_texte = ""       # tout le texte diffusé token par token
    en_cours = False      # une ligne de réponse est en train d'être écrite
    interruption = None

    async for mode, chunk in agent.astream(
        entree, config, stream_mode=["messages", "updates"]
    ):
        # --- Tokens de la réponse ---
        if mode == "messages":
            message, meta = chunk
            if meta.get("langgraph_node") != "model":
                continue
            if isinstance(message, AIMessageChunk):
                morceau = texte(message.content)
                if morceau:
                    if not en_cours:
                        print("💬 ", end="", flush=True)
                        en_cours = True
                    print(morceau, end="", flush=True)
                    flux_texte += morceau
            continue

        # --- Étapes du graphe ---
        if "__interrupt__" in chunk:
            interruption = chunk["__interrupt__"][0].value
            continue

        for _noeud, maj in chunk.items():
            for msg in (maj or {}).get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    if en_cours:
                        print()
                        en_cours = False
                    for appel in msg.tool_calls:
                        print(f"🔧 {appel['name']}({court(appel['args'])})")
                elif isinstance(msg, ToolMessage):
                    print(f"   ↳ {court(texte(msg.content) or msg.content)}")

    if en_cours:
        print()

    if interruption is not None:
        return interruption

    # À la reprise après une interruption, LangGraph ne rediffuse pas les
    # tokens : on lit la réponse finale dans l'état sauvegardé.
    etat = await agent.aget_state(config)
    messages = etat.values.get("messages", [])
    final = texte(messages[-1].content).strip() if messages else ""
    if final and final not in flux_texte:
        print(f"💬 {final}")

    return None


# ============================================================
# BOUCLE DE CHAT
# ============================================================

async def chat(thread_id: str):
    async with AsyncSqliteSaver.from_conn_string(DB) as checkpointer:
        # Une seule session MCP pour toute la conversation
        async with open_agent(checkpointer) as agent:
            config = {"configurable": {"thread_id": thread_id}}

            print("\n" + "=" * 60)
            print("  Assistant de support Randoneo")
            print(f"  Conversation : {thread_id}")
            print("  (relancez avec --thread <id> pour la reprendre)")
            print("  Tapez 'quit' pour quitter.")
            print("=" * 60 + "\n")

            while True:
                try:
                    question = input("❓ Vous : ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 À bientôt !")
                    break

                if question.lower() in ("quit", "exit", "q"):
                    print("👋 À bientôt !")
                    break
                if not question:
                    continue

                print()
                try:
                    entree = {"messages": [HumanMessage(question)]}
                    demande = await tour(agent, entree, config)

                    # Tant que le graphe attend une validation, on la demande
                    while demande is not None:
                        approuve = demander_validation(demande)
                        print()
                        demande = await tour(
                            agent, Command(resume={"approved": approuve}), config
                        )
                except Exception as e:
                    print(f"\n❌ Erreur : {e}")

                print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assistant de support Randoneo")
    parser.add_argument(
        "--thread",
        default=None,
        help="identifiant de conversation à reprendre (nouveau par défaut)",
    )
    args = parser.parse_args()

    asyncio.run(chat(args.thread or f"session-{uuid.uuid4().hex[:8]}"))