"""
Agent Randoneo — sans aucun outil en dur.

Les capacités viennent du serveur MCP (randoneo_support.py).
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
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

# Le client MCP (stdio)
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
    """Charge les tools du serveur MCP et construit l'agent avec mémoire."""
    # 1. Charge les tools depuis le serveur MCP
    tools = await client.get_tools()
    print(f"✅ {len(tools)} tools chargés depuis le MCP :")
    for tool in tools:
        print(f"   - {tool.name}")
    
    # 2. Crée le modèle
    model = init_chat_model(
        "claude-haiku-4-5",
        model_provider="anthropic",
        temperature=0,
        max_retries=8,
    )
    
    # 3. Crée l'agent avec checkpointer
    agent = create_agent(
        model,
        tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    
    return agent


# ============================================================
# TEST
# ============================================================

async def main():
    DB = "randoneo_memory.sqlite"
    
    # Ouvre la base SQLite (asynchrone)
    async with AsyncSqliteSaver.from_conn_string(DB) as checkpointer:
        agent = await build_agent(checkpointer)
        
        print("\n" + "=" * 60)
        print("Test de la mémoire")
        print("=" * 60 + "\n")
        
        # Configuration du thread
        config = {"configurable": {"thread_id": "camille-1"}}
        
        # Test 1 : première question
        print("❓ Test 1 : Où en est ma commande RND-10235 ?")
        result = await agent.ainvoke(
            {"messages": [HumanMessage("Où en est ma commande RND-10235 ?")]},
            config,
        )
        print(f"💬 {result['messages'][-1].content}\n")
        
        # Test 2 : question de suivi (sans répéter le numéro)
        print("❓ Test 2 : Et elle arrive quand, exactement ?")
        result = await agent.ainvoke(
            {"messages": [HumanMessage("Et elle arrive quand, exactement ?")]},
            config,
        )
        print(f"💬 {result['messages'][-1].content}\n")


if __name__ == "__main__":
    asyncio.run(main())