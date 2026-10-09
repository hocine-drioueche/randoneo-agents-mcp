"""
Rejoue sans interaction les six scénarios de vérification de l'assistant Randoneo.

Chaque scénario vérifie les outils réellement appelés (et non seulement le
texte de la réponse). Les validations humaines sont simulées : approbation ou
refus codés en dur.

Lancement : python demo.py
"""

import asyncio
import sys
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from agent import open_agent

DB = "randoneo_memory.sqlite"
RUN = uuid.uuid4().hex[:6]  # threads neufs à chaque exécution


def texte(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


# ============================================================
# EXÉCUTION D'UN TOUR (sans affichage de tokens)
# ============================================================

async def tour(agent, entree, config, trace):
    """Fait avancer l'agent ; alimente `trace` (outils appelés, résultats).

    Retourne la demande de validation si le graphe s'est interrompu, sinon None.
    """
    interruption = None
    async for chunk in agent.astream(entree, config, stream_mode="updates"):
        if "__interrupt__" in chunk:
            interruption = chunk["__interrupt__"][0].value
            continue
        for maj in chunk.values():
            for msg in (maj or {}).get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    trace["tools"] += [c["name"] for c in msg.tool_calls]
                elif isinstance(msg, ToolMessage):
                    trace["results"].append(texte(msg.content) or str(msg.content))
    return interruption


async def jouer(agent, thread_id, question, decision=None):
    """Joue un scénario. `decision` = None (pas d'interruption attendue),
    True (approuver) ou False (refuser)."""
    config = {"configurable": {"thread_id": thread_id}}
    trace = {"tools": [], "results": [], "interrupted": None}

    demande = await tour(agent, {"messages": [HumanMessage(question)]}, config, trace)
    trace["interrupted"] = demande

    while demande is not None:
        approuve = bool(decision)
        print(f"   ⏸️  validation demandée : {demande['action']}({demande['args']})"
              f" → opérateur : {'OUI' if approuve else 'NON'}")
        demande = await tour(agent, Command(resume={"approved": approuve}), config, trace)

    etat = await agent.aget_state(config)
    trace["answer"] = texte(etat.values["messages"][-1].content).strip()
    return trace


# ============================================================
# SCÉNARIOS
# ============================================================

async def main() -> int:
    resultats = []

    def verifier(nom, conditions, trace):
        ok = all(c for _, c in conditions)
        resultats.append(ok)
        print(f"   🔧 outils appelés : {trace['tools'] or 'aucun'}")
        for libelle, c in conditions:
            print(f"   {'✅' if c else '❌'} {libelle}")
        print(f"   💬 {trace['answer'][:400]}")
        print(f"   ==> {'PASS' if ok else 'FAIL'}\n")

    async with AsyncSqliteSaver.from_conn_string(DB) as checkpointer:
        async with open_agent(checkpointer) as agent:

            # 1 — retour : search_knowledge_base seul
            print("\n① J'ai combien de temps pour retourner un article ?")
            t = await jouer(agent, f"demo-{RUN}-1", "J'ai combien de temps pour retourner un article ?")
            verifier("1", [
                ("search_knowledge_base appelé", "search_knowledge_base" in t["tools"]),
                ("aucun autre outil", set(t["tools"]) == {"search_knowledge_base"}),
            ], t)

            # 2 — suivi de commande : get_order_status seul (thread réutilisé en ⑥)
            thread_suivi = f"demo-{RUN}-2"
            print("② Où en est ma commande RND-10234 ?")
            t = await jouer(agent, thread_suivi, "Où en est ma commande RND-10234 ?")
            verifier("2", [
                ("get_order_status appelé", "get_order_status" in t["tools"]),
                ("aucun autre outil", set(t["tools"]) == {"get_order_status"}),
            ], t)

            # 3 — les deux outils, puis réponse combinée
            print("③ Où en est RND-10238, et sous quel délai serai-je remboursé ?")
            t = await jouer(agent, f"demo-{RUN}-3",
                            "Où en est RND-10238, et sous quel délai serai-je remboursé ?")
            verifier("3", [
                ("get_order_status appelé", "get_order_status" in t["tools"]),
                ("search_knowledge_base appelé", "search_knowledge_base" in t["tools"]),
                ("la réponse mentionne le délai (14 jours)", "14" in t["answer"]),
            ], t)

            # 4 — action sensible approuvée : interruption puis ticket ouvert
            print("④ Tente déchirée, remplacement demandé — opérateur : OUI")
            t = await jouer(agent, f"demo-{RUN}-4",
                            "Ma tente de RND-10238 est arrivée déchirée, je veux un remplacement",
                            decision=True)
            verifier("4", [
                ("interruption avant exécution", t["interrupted"] is not None),
                ("create_ticket appelé", "create_ticket" in t["tools"]),
                ("ticket réellement ouvert (TICK-…)", any("TICK-" in r for r in t["results"])),
            ], t)

            # 5 — même demande, refusée par l'opérateur
            print("⑤ Même demande — opérateur : NON")
            t = await jouer(agent, f"demo-{RUN}-5",
                            "Ma tente de RND-10238 est arrivée déchirée, je veux un remplacement",
                            decision=False)
            verifier("5", [
                ("interruption avant exécution", t["interrupted"] is not None),
                ("aucun ticket créé", not any("TICK-" in r for r in t["results"])),
                ("refus signalé à l'agent", any("refusée" in r for r in t["results"])),
            ], t)

            # 6 — question de suivi sur le thread du scénario 2 : la mémoire joue
            print("⑥ Et elle arrive quand ? (thread du scénario ②)")
            t = await jouer(agent, thread_suivi, "Et elle arrive quand ?")
            verifier("6", [
                ("aucun outil rappelé", not t["tools"]),
                ("la réponse contient la date d'arrivée (15 juin)",
                 "15" in t["answer"] and "juin" in t["answer"].lower()
                 or "2026-06-15" in t["answer"]),
            ], t)

    print("=" * 50)
    print(f"Résultat : {sum(resultats)}/{len(resultats)} scénarios validés")
    return 0 if all(resultats) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))