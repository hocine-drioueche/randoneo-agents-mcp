"""
Agent Randoneo — sans aucun outil métier en dur.

Les capacités viennent du serveur MCP (randoneo_support.py), via UNE session
ouverte pour toute la durée de la conversation (le serveur ne redémarre pas
à chaque appel d'outil : base vectorielle et tickets restent en mémoire).

Le garde-fou humain (interrupt) est posé ici, côté agent : on enveloppe
les tools sensibles découverts avant de les donner au modèle.
"""

import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command, interrupt

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

SYSTEM_PROMPT = (
    "Tu es l'assistant du support client de Randoneo, un vendeur de matériel "
    "de randonnée. Utilise tes outils pour renseigner le client. "
    "Réponds en français, de façon concise et professionnelle. "
    "Quand le client demande une action (remplacement, remboursement, "
    "réclamation) et fournit un numéro de commande, ouvre directement un "
    "ticket avec create_ticket : un opérateur humain valide chaque ticket, "
    "ne pose pas de questions de clarification superflues. "
    "N'invente jamais de fonctionnalité ni de délai : n'annonce que ce que "
    "renvoient tes outils (pas d'email de confirmation, pas de notification "
    "à moins que la documentation le dise). "
    "Si une action est refusée par l'opérateur, annonce clairement au client "
    "qu'aucune demande n'a été enregistrée, sans réessayer."
)

# Tools du serveur MCP qui agissent réellement : validation humaine obligatoire.
TOOLS_SENSIBLES = {"create_ticket"}

client = MultiServerMCPClient({
    "randoneo": {
        "command": "python",
        "args": ["randoneo_support.py"],
        "transport": "stdio",
        "env": {
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"],
            "PATH": os.environ["PATH"],
        },
    }
})


# ============================================================
# GARDE-FOU HUMAIN
# ============================================================

def avec_validation_humaine(mcp_tool):
    """Enveloppe un tool MCP découvert : demande le feu vert avant de déléguer.

    Le wrapper reprend nom, description et schéma du tool d'origine : le modèle
    ne voit aucune différence. La décision vit côté agent (interrupt est une
    primitive du graphe), l'action reste côté serveur.
    """

    async def _garde(**kwargs):
        decision = interrupt({
            "action": mcp_tool.name,
            "args": kwargs,
        })

        # decision = {"approved": bool}, fourni par chat.py via Command(resume=...)
        if isinstance(decision, dict) and decision.get("approved"):
            return await mcp_tool.ainvoke(kwargs)

        return (
            f"Action « {mcp_tool.name} » refusée par l'opérateur humain : "
            "aucun ticket n'a été créé. Informe le client qu'un conseiller "
            "n'a pas validé la demande ; ne propose pas de rouvrir un ticket "
            "toi-même."
        )

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.args_schema,
        coroutine=_garde,
    )


# ============================================================
# CONSTRUCTION DE L'AGENT (session MCP persistante)
# ============================================================

@asynccontextmanager
async def open_agent(checkpointer):
    """Ouvre UNE session MCP, charge les tools, protège les sensibles.

    À utiliser avec `async with open_agent(checkpointer) as agent:`.
    La session (et donc le processus serveur) reste ouverte tant que le bloc
    `async with` est actif, puis est fermée proprement à la sortie.
    """
    async with client.session("randoneo") as session:
        mcp_tools = await load_mcp_tools(session)

        tools = [
            avec_validation_humaine(t) if t.name in TOOLS_SENSIBLES else t
            for t in mcp_tools
        ]

        print(f"✅ {len(tools)} tools chargés depuis le MCP")
        for t in tools:
            marque = " 🔒 (validation humaine)" if t.name in TOOLS_SENSIBLES else ""
            print(f"   - {t.name}{marque}")

        model = init_chat_model(
            "claude-haiku-4-5",
            model_provider="anthropic",
            temperature=0,
            max_retries=8,
        )

        yield create_agent(
            model,
            tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )


# ============================================================
# TEST RAPIDE DU GARDE-FOU (sans chat.py)
# ============================================================

async def _tour(agent, entree, config):
    """Un tour : renvoie (interruption | None, réponse finale | None)."""
    result = await agent.ainvoke(entree, config)
    interruptions = result.get("__interrupt__")
    if interruptions:
        return interruptions[0].value, None
    return None, result["messages"][-1].content


async def main():
    async with AsyncSqliteSaver.from_conn_string("randoneo_memory.sqlite") as checkpointer:
        async with open_agent(checkpointer) as agent:
            demande = "Ma tente de RND-10238 est arrivée déchirée, je veux un remplacement"

            for approuve, thread in [(True, "test-garde-fou-oui"), (False, "test-garde-fou-non")]:
                config = {"configurable": {"thread_id": thread}}
                print(f"\n❓ {demande}  (opérateur : {'OUI' if approuve else 'NON'})")

                demande_validation, reponse = await _tour(
                    agent, {"messages": [HumanMessage(demande)]}, config
                )
                if demande_validation:
                    print(f"⏸️  Validation requise : {demande_validation}")
                    _, reponse = await _tour(
                        agent, Command(resume={"approved": approuve}), config
                    )
                print(f"💬 {reponse}")


if __name__ == "__main__":
    asyncio.run(main())