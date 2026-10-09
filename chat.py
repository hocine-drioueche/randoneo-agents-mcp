"""
Boucle de chat interactive pour l'assistant Randoneo.

- Streaming des étapes
- Garde-fou humain (géré manuellement)
- Mémoire persistante (SQLite)
"""

import asyncio
import os
import uuid

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Charge le .env
load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

SYSTEM_PROMPT = (
    "Tu es l'assistant du support client de Randoneo, un vendeur de matériel "
    "de randonnée. Utilise tes outils pour renseigner le client. "
    "Réponds en français, de façon concise et professionnelle."
)

# Tools qui nécessitent une validation humaine
SENSITIVE_TOOLS = {"create_ticket"}

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
# CONSTRUCTION DE L'AGENT
# ============================================================

async def build_agent(checkpointer):
    """Charge les tools du serveur MCP et construit l'agent."""
    mcp_tools = await client.get_tools()
    print(f"✅ {len(mcp_tools)} tools chargés depuis le MCP")
    
    model = init_chat_model(
        "claude-haiku-4-5",
        model_provider="anthropic",
        temperature=0,
        max_retries=8,
    )
    
    agent = create_agent(
        model,
        mcp_tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    
    return agent, mcp_tools


# ============================================================
# GARDE-FOU HUMAIN (manuel)
# ============================================================

async def handle_sensitive_tools(agent, config, state):
    """
    Vérifie si le dernier message contient un tool_use sensible.
    Si oui, demande validation à l'opérateur.
    """
    last_message = state.values["messages"][-1]
    
    # Vérifie s'il y a un tool_use
    if not isinstance(last_message, AIMessage):
        return None
    
    if not last_message.tool_calls:
        return None
    
    # Cherche un tool sensible
    sensitive_calls = [
        call for call in last_message.tool_calls
        if call["name"] in SENSITIVE_TOOLS
    ]
    
    if not sensitive_calls:
        return None
    
    # Affiche la demande de validation
    call = sensitive_calls[0]
    print("\n" + "=" * 60)
    print("  ⚠️  VALIDATION HUMAINE REQUISE")
    print("=" * 60)
    print(f"  Action : {call['name']}")
    print(f"  Arguments :")
    for key, value in call["args"].items():
        print(f"    - {key} : {value}")
    print("=" * 60)
    
    # Demande la décision
    while True:
        decision = input("\n  Approuver ? (o/n) : ").strip().lower()
        if decision in ("o", "oui", "y", "yes"):
            approved = True
            break
        elif decision in ("n", "non", "no"):
            approved = False
            break
        print("  Répondez par 'o' ou 'n'.")
    
    return approved


# ============================================================
# BOUCLE DE CHAT
# ============================================================

async def chat():
    DB = "randoneo_memory.sqlite"
    
    async with AsyncSqliteSaver.from_conn_string(DB) as checkpointer:
        agent, mcp_tools = await build_agent(checkpointer)
        
        print("\n" + "=" * 60)
        print("  Assistant de support Randoneo")
        print("  Tapez 'quit' pour quitter.")
        print("=" * 60 + "\n")
        
        # Un thread_id par session
        thread_id = f"session-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Dictionnaire des tools pour exécution manuelle
        tools_dict = {t.name: t for t in mcp_tools}
        
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
                # Lance l'agent
                result = await agent.ainvoke(
                    {"messages": [HumanMessage(question)]},
                    config,
                )
                
                # Boucle tant qu'il y a des tool_use à traiter
                while True:
                    # Récupère l'état
                    state = await agent.aget_state(config)
                    
                    # Vérifie s'il y a un tool_use sensible
                    approved = await handle_sensitive_tools(agent, config, state)
                    
                    if approved is None:
                        # Pas de tool sensible → terminé
                        break
                    
                    # Récupère le dernier message
                    last_message = state.values["messages"][-1]
                    sensitive_call = next(
                        call for call in last_message.tool_calls
                        if call["name"] in SENSITIVE_TOOLS
                    )
                    
                    # Exécute ou refuse
                    if approved:
                        print("\n  ✅ Approuvé. Exécution en cours...")
                        tool = tools_dict[sensitive_call["name"]]
                        tool_result = await tool.ainvoke(sensitive_call["args"])
                        tool_message = ToolMessage(
                            content=str(tool_result),
                            tool_call_id=sensitive_call["id"],
                        )
                    else:
                        print("\n  ❌ Refusé.")
                        tool_message = ToolMessage(
                            content="Action refusée par un opérateur.",
                            tool_call_id=sensitive_call["id"],
                        )
                    
                    # Injecte le tool_result
                    await agent.aupdate_state(
                        config,
                        {"messages": [tool_message]},
                    )
                    
                    # Reprend l'agent
                    result = await agent.ainvoke(None, config)
                
                # Affiche la réponse finale
                final_state = await agent.aget_state(config)
                final_message = final_state.values["messages"][-1]
                
                print("\n" + "-" * 60)
                print(f"💬 {final_message.content}")
                print("-" * 60 + "\n")
            
            except Exception as e:
                print(f"\n❌ Erreur : {e}\n")


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    asyncio.run(chat())