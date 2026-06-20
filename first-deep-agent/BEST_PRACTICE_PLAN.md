# Best Practice Plan

## Zielbild

Der Agent Server bleibt ein offizielles LangGraph/LangSmith Deployment. Deep Agents ist die Agent-Harness-Schicht fuer Planung, Subagents, Dateien und Memory. Die Chat UI bleibt getrennt und spricht ueber einen serverseitigen Proxy mit dem Agent Server.

## Reihenfolge

1. Agent-Basis stabilisieren
   - System Prompt, Subagents und Tool-Freigaben klar halten.
   - Schreibende Dateioperationen per Interrupt pruefbar machen.
   - Keine direkte Host-Dateisystem-Freigabe im Webserver verwenden.

2. Memory sauber trennen
   - Thread-Zustand bleibt im Checkpointer.
   - Kurzlebige Arbeitsdateien bleiben im StateBackend.
   - Langzeitnotizen liegen nur unter `/memories/` im StoreBackend.
   - Secrets duerfen nie in Memory oder Dateien gespeichert werden.

3. Deployment-Konfiguration haerten
   - LangSmith Deployment stellt Store und Checkpointer bereit.
   - TTL-Regeln verhindern unbegrenztes Wachstum.
   - Semantic Search erst aktivieren, wenn das Embedding-Modell feststeht.

4. MCP gezielt anbinden
   - Nur MCP-Server laden, die fuer den Agenten wirklich gebraucht werden.
   - Schreibende oder riskante Tools brauchen Freigabe.
   - OAuth- oder Header-Auth bleibt serverseitig.

5. RAG gezielt bauen
   - Kleine lokale RAG-Strecke zuerst.
   - Danach entscheiden: LangGraph Store Search, Chroma, Qdrant, Pinecone oder Weaviate.

6. Auth vor echter Mehrnutzer-Nutzung
   - Spaeter Custom Auth ergaenzen.
   - Memory-Namespace dann pro User oder Projekt isolieren.
   - UI-Proxy nicht als einzige Sicherheitsgrenze fuer Public-Zugriff verwenden.

7. Production-Review
   - Unbenutzte Provider-Pakete entfernen.
   - Kosten, Traces, Latenz und Tool-Nutzung in LangSmith pruefen.
   - Erst danach ueber eigenes VPS/Self-hosting entscheiden.

## Aktuelle Entscheidung

Managed Deep Agents ist langfristig interessant, aber fuer dieses Setup noch nicht der primaere Weg, weil wir EU-Workspace, Custom-Code, Agent-Server-API, eigene UI und spaetere Auth/MCP/RAG-Erweiterungen brauchen. Der beste aktuelle Weg ist daher: LangSmith/LangGraph Deployment plus Deep Agents im eigenen Graph.
