"""
Boucle de chat interactive pour l'assistant Randoneo.

- Streaming des étapes
- Garde-fou humain (interrupt)
- Mémoire persistante (SQLite)
"""

import asyncio
import os
import uuid

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
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
    "Si une action est refusée par l'opérateur, annonce clairement au client "
    "qu'aucune demande n'a été enregistrée, sans réessayer."
)

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
    """Enveloppe un tool MCP découvert : demande le feu vert avant de déléguer."""
    async def _garde(**kwargs):
        decision = interrupt({
            "action": mcp_tool.name,
            "args": kwargs,
        })
        
        if isinstance(decision, dict) and decision.get("approved"):
            return await mcp_tool.ainvoke(kwargs)
        
        return (
            f"Action « {mcp_tool.name} » refusée par l'opérateur humain : "
            "aucun ticket n'a été créé."
        )
    
    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.args_schema,
        coroutine=_garde,
    )


# ============================================================
# CONSTRUCTION DE L'AGENT
# ============================================================

async def build_agent(checkpointer):
    """Charge les tools du serveur MCP, protège les sensibles, construit l'agent."""
    mcp_tools = await client.get_tools()
    
    tools = [
        avec_validation_humaine(t) if t.name in TOOLS_SENSIBLES else t
        for t in mcp_tools
    ]
    
    print(f"✅ {len(tools)} tools chargés")
    for t in tools:
        marque = " 🔒" if t.name in TOOLS_SENSIBLES else ""
        print(f"   - {t.name}{marque}")
    
    model = init_chat_model(
        "claude-haiku-4-5",
        model_provider="anthropic",
        temperature=0,
        max_retries=8,
    )
    
    return create_agent(
        model,
        tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


# ============================================================
# BOUCLE DE CHAT
# ============================================================

async def chat():
    DB = "randoneo_memory.sqlite"
    
    async with AsyncSqliteSaver.from_conn_string(DB) as checkpointer:
        agent = await build_agent(checkpointer)
        
        print("\n" + "=" * 60)
        print("  Assistant de support Randoneo")
        print("  Tapez 'quit' pour quitter.")
        print("=" * 60 + "\n")
        
        thread_id = f"session-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        
        while True:
            try:
                question = input("❓ Votre question : ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 À bientôt !")
                break
            
            if question.lower() in ("quit", "exit", "q"):
                print("👋 À bientôt !")
                break
            
            if not question:
                continue
            
            print()
            
            try:
                # Boucle : on relance tant qu'il y a des interruptions
                entree = {"messages": [HumanMessage(question)]}
                
                while True:
                    result = await agent.ainvoke(entree, config)
                    
                    # Vérifie s'il y a une interruption
                    interruptions = result.get("__interrupt__")
                    
                    if not interruptions:
                        # Pas d'interruption → réponse finale
                        print("-" * 60)
                        print(f"💬 {result['messages'][-1].content}")
                        print("-" * 60 + "\n")
                        break
                    
                    # Interruption → demande validation
                    interrupt_data = interruptions[0].value
                    print("\n" + "=" * 60)
                    print("  ⚠️  VALIDATION HUMAINE REQUISE")
                    print("=" * 60)
                    print(f"  Action : {interrupt_data['action']}")
                    print(f"  Arguments :")
                    for key, value in interrupt_data["args"].items():
                        print(f"    - {key} : {value}")
                    print("=" * 60)
                    
                    # Demande décision
                    while True:
                        decision = input("\n  Approuver ? (o/n) : ").strip().lower()
                        if decision in ("o", "oui", "y", "yes"):
                            approved = True
                            break
                        elif decision in ("n", "non", "no"):
                            approved = False
                            break
                        print("  Répondez par 'o' ou 'n'.")
                    
                    print()
                    entree = Command(resume={"approved": approved})
            
            except Exception as e:
                print(f"\n❌ Erreur : {e}\n")


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    asyncio.run(chat())